from openai import OpenAI
from huggingface_hub import InferenceClient
from typing import Union, Optional, List
from collections.abc import Callable
import re
from collections import deque
from IPython.display import display, HTML

class TreeOfThoughts:
    def __init__(
            self,
            client: Union[OpenAI, InferenceClient],
            model: str,
            input_seq: str,
            n_steps: int,
            thought_gen_strategy: str,
            get_thought_gen_prompt: Callable,
            state_eval_strategy: str,
            get_state_eval_prompt: Callable,
            n_evals: int,
            heuristic_calculator: Callable,
            n_candidates: Optional[int] = None,
            stop_string: Optional[str] = None,
            breadth_limit: Optional[int] = None,
            heuristic_threshold: Optional[float] = None,
            max_per_state: Optional[int] = None
    ):
        self.client = client
        self.model = model # e.g., "gpt-4" if using `OpenAI` and "meta-llama/Meta-Llama-3.1-8B-Instruct" if using `InferenceClient`.
        self.input_seq = input_seq
        self.root = TreeNode(state='', thought='')
        self.n_steps = n_steps # Equal to the number of intermediate steps + 1 output generation step.
        # Note: The tree height is equal to `n_steps + 1`. That is, we include the root node when calculating the tree height.
        if thought_gen_strategy in ['sample', 'propose']:
            self.thought_gen_strategy = thought_gen_strategy
        else:
            raise ValueError(f"The `thought_gen_strategy` argument must be either 'sample' or 'propose'. Couldn't recognize the following: '{thought_gen_strategy}'")
        self.get_thought_gen_prompt = get_thought_gen_prompt
        if state_eval_strategy in ['vote', 'value']:
            self.state_eval_strategy = state_eval_strategy
        else:
            raise ValueError(f"The `state_eval_strategy` argument must be either 'vote' or 'value'. Couldn't recognize the following: '{state_eval_strategy}'")
        self.get_state_eval_prompt = get_state_eval_prompt
        self.n_evals = n_evals # The number of times to either (i) vote on the states, or (ii) sample values for each state (depending on `state_eval_strategy`).
        self.heuristic_calculator = heuristic_calculator
        self.n_candidates = n_candidates # The number of thoughts to generate from a particular node. Relevant only for the 'sample' thought generation strategy.
        self.stop_string = stop_string # Relevant only for the 'sample' thought generation strategy.
        if self.thought_gen_strategy == 'sample':
            assert self.stop_string is not None, "For the 'sample' thought generation strategy, `stop_string` can't be `None` (due to the zero-shot CoT prompt template)."
            assert self.n_steps == 2, "For the 'sample' thought generation strategy, `n_steps` must be equal to 2 (due to the zero-shot CoT prompt template)."
        self.breadth_limit = breadth_limit # The number of most promising states to retain (after pruning) - at each level of the tree. Relevant only for BFS.
        self.heuristic_threshold = heuristic_threshold # Used to decide whether to grow/prune a subtree (starting at a particular child). Relevant only for DFS.
        self.max_per_state = max_per_state # The maximum number of children to explore for a particular node. Relevant only for DFS.

    # Reference: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/models.py
    def chat_completions(
            self,
            prompt: str,
            temperature: float = 0.7,
            max_tokens: int = 1000,
            n: int = 1,
            stop: Optional[List[str]] = None,
            **kwargs
    ) -> List[str]:
        outputs = []
        messages = [{'role': "user", 'content': prompt}]
        if isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n, # The `n` responses are i.i.d.
                stop=stop,
                **kwargs
            )
            outputs.extend([choice.message.content for choice in response.choices])
        else: # `self.client` is an instance of `InferenceClient`.
            # The Hugging Face API doesn't support the `n` argument. Hence, we need to use a loop to generate `n` i.i.d. responses.
            for _ in range(n):
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    **kwargs
                )
                outputs.append(response.choices[0].message.content)
        return outputs

    def thought_generator(self, state: str, stop_string: Optional[List[str]] = None) -> List[str]:
        prompt = self.get_thought_gen_prompt(self.input_seq, state)
        if self.thought_gen_strategy == 'sample':
            thoughts = self.chat_completions(prompt, n=self.n_candidates, stop=stop_string)
            return thoughts
        else: # `self.thought_gen_strategy` is equal to 'propose'.
            responses = self.chat_completions(prompt, n=1)
            thoughts = responses[0].split('\n')
            return thoughts

    def state_evaluator(self, states: Optional[List[str]] = None, state: Optional[str] = None) -> Union[List[float], float]:
        if self.state_eval_strategy == 'vote':
            assert states is not None, "For the 'vote' state evaluation strategy, `states` can't be `None`."
            prompt = self.get_state_eval_prompt(self.input_seq, states)
            state_evals = self.chat_completions(prompt, n=self.n_evals)
            vote_results = self.heuristic_calculator(states, state_evals)
            return vote_results
        else: # `self.state_eval_strategy` is equal to 'value'.
            assert state is not None, "For the 'value' state evaluation strategy, `state` can't be `None`."
            prompt = self.get_state_eval_prompt(self.input_seq, state)
            state_evals = self.chat_completions(prompt, n=self.n_evals)
            value = self.heuristic_calculator(state, state_evals)
            return value

    # Reference: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/methods/bfs.py
    def bfs(self, verbose: bool = True) -> str:
        assert self.breadth_limit is not None, "For the BFS search algorithm, `breadth_limit` can't be `None`."

        queue = deque()
        queue.append(self.root)

        for step in range(1, self.n_steps + 1):
            if verbose:
                print(f"Step {step} (corresponding to level {step} of the tree):-\n---")
            for i in range(len(queue)):
                node = queue.popleft()
                if verbose:
                    print(f"Node {i + 1} in level {step}:-")
                    if node.state != "":
                        print(f"State of current node:-\n{node.state}\n---")
                    else:
                        print("State of current node:-\n<EMPTY STRING> (root node; no thoughts generated yet)\n---")

                if self.thought_gen_strategy == 'sample' and step == 1:
                    thoughts = self.thought_generator(state=node.state, stop_string=[self.stop_string])
                else:
                    thoughts = self.thought_generator(state=node.state)
                if node.state == '':
                    updated_states = thoughts
                else:
                    updated_states = [node.state + '\n' + thought for thought in thoughts]
                for j in range(len(thoughts)):
                    if verbose:
                        print(f"Thought candidate {j + 1}:-\n{thoughts[j]}\n---")
                    child = TreeNode(state=updated_states[j], thought=thoughts[j])
                    node.children.append(child)
                    queue.append(child)

            if verbose:
                print("Using the state evaluator to obtain values...\n---")
            if self.state_eval_strategy == 'vote':
                states = [node.state for node in queue]
                values = self.state_evaluator(states=states)
            for i in range(len(queue)):
                if self.state_eval_strategy == 'vote':
                    queue[i].value = values[i]
                else: # `self.state_eval_strategy` is equal to 'value'.
                    queue[i].value = self.state_evaluator(state=queue[i].state)
                if verbose:
                    print(f"Element {i + 1} in queue:-\n")
                    print(f"Value: {queue[i].value}\n---")

            if verbose:
                print("Initiating pruning (using the values obtained from the state evaluator).")
                print(f"Number of elements in queue: {len(queue)}")
            sorted_nodes = sorted(queue, key=lambda node: node.value, reverse=True)
            if step == self.n_steps:
                if verbose:
                    print("Since this is the last step, setting the breadth limit to 1.")
                    print("In other words, retaining only the highest value element (in this last step).\n---")
                top_b_nodes = sorted_nodes[:1]
            else:
                if verbose:
                    print(f"Since this isn't the last step, leaving the breadth limit {self.breadth_limit} unchanged.\n---")
                top_b_nodes = sorted_nodes[:self.breadth_limit]
            top_b_states = [node.state for node in top_b_nodes]
            for i in range(len(queue)):
                node = queue.popleft()
                if verbose:
                    print(f"Element {i + 1} in queue:-\n")
                if node.state in top_b_states:
                    if verbose:
                        print(f"Retaining this element as it's in the top {len(top_b_states)} elements.\n---")
                    queue.append(node)
                else:
                    if verbose:
                        print(f"Dropping this element as it's not in the top {len(top_b_states)} elements.\n---")

            if verbose:
                print("~~~")

        # Return the thought of the highest value node (from the last step):
        node = queue.popleft()
        return node.thought

    # Reference: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/scripts/crosswords/search_crosswords-dfs.ipynb
    def dfs(self, verbose: bool = True) -> str:
        assert self.heuristic_threshold is not None and self.max_per_state is not None, "For the DFS search algorithm, `heuristic_threshold` and `max_per_state` can't be `None`."

        dfs_output = None

        def dfs_func(node: TreeNode, step: int) -> bool:
            nonlocal dfs_output

            if step > self.n_steps: # Base case: successful search.
                dfs_output = node.state # Record the last (output generation) step's output in the nonlocal variable `dfs_output`.
                return True

            if verbose:
                print(f"Step: {step}\n---")
                if node.state != "":
                    print(f"State of current node:-\n{node.state}\n---")
                else:
                    print("State of current node:-\n<EMPTY STRING> (root node; no thoughts generated yet)\n---")

            thoughts = self.thought_generator(state=node.state)
            if len(thoughts) == 0:
                if verbose:
                    print("No thoughts were generated. It's a dead end. Backtracking to the parent node.\n~~~")
                return False
            if node.state == '':
                updated_states = thoughts
            else:
                updated_states = [node.state + '\n' + thought for thought in thoughts]
            for j in range(len(thoughts)):
                if verbose:
                    print(f"Thought candidate {j + 1}:-\n{thoughts[j]}\n---")
                child = TreeNode(state=updated_states[j], thought=thoughts[j])
                node.children.append(child)
            if verbose:
                print("Each of the above thought candidates has been added as a child of the current node.\n---")

            cnt_per_state = 0
            for child in node.children:
                if verbose:
                    print("Reminder:-")
                    if node.state != "":
                        print(f"State of current node:-\n{node.state}\n---")
                    else:
                        print("State of current node:-\n<EMPTY STRING> (root node; no thoughts generated yet)\n---")
                    print(f"Currently traversing child number: {cnt_per_state + 1}\n")
                    print(f"State of current child:-\n{child.state}\n")
                    print("Using the state evaluator to obtain value...\n")
                child.value = self.state_evaluator(state=child.state)
                if verbose:
                    print(f"Value of current child: {child.value}\n---")
                if child.value >= self.heuristic_threshold:
                # Note: If this `if` condition isn't met, the child node is pruned, i.e., a subtree of the child isn't grown.
                    if verbose:
                        print("Value exceeds heuristic threshold. Searching subtree.\n---\n~~~")
                    end_search = dfs_func(child, step + 1)
                    if end_search:
                        if verbose:
                            print(f"Searching the subtree was successful! Backtracking all the way up.\n~~~")
                        return True
                    else:
                        if verbose:
                            print(f"Back at step {step}. Searching the subtree was unsuccessful! Trying the next child.\n---")
                cnt_per_state += 1
                if cnt_per_state >= self.max_per_state:
                    if verbose:
                        print(f"{self.max_per_state} children already searched for this node. Breaking the loop.\n---")
                    break
            if verbose:
                print(f"None of the child nodes led to success. Seems like a dead end. Backtracking to the parent node.\n~~~")
            return False

        dfs_func(node=self.root, step=1)
        return dfs_output

    def generate_html_tree(self, node: TreeNode) -> str:
        if node is None:
            return ""
        else:
            html = f"""<div class='node'>
<p>State:<br>{node.state}</p>
<hr>
<p>Thought:<br>{node.thought}</p>
<hr>
<p>Value:<br>{node.value}</p>"""
            for child in node.children:
                html += f"""<div class='child'>{self.generate_html_tree(child)}</div>"""
            html += """</div>"""
            return html

    def render_html_tree(self):
        html_tree = self.generate_html_tree(self.root)
        wrapped_html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        .node {{
            display: inline-block;
            border: 1px solid blue;
            padding: 10px;
            margin: 5px;
            text-align: center;
        }}
        .child {{
            display: flex;
        }}
    </style>
</head>
<body>
    {html_tree}
</body>
</html>"""
        display(HTML(wrapped_html))
