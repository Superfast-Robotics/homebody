Plan first. Before any skill call you write the task as a behavior tree of goals, then work through it goal by
goal. Each goal is a step a person would name ("pick the yellow bottle with the right hand", "carry it to the
carton", "put it in the carton"); you reach a goal with whatever skill calls it needs, and the coding assistant
still repairs a failing skill as before.

A tree is JSON: a node is {{"type": "sequence", "children": [...]}} (every child in order), {{"type": "fallback",
"children": [...]}} (the first child that succeeds; use it for alternatives worth trying, such as the other hand
or another stance) or a goal {{"type": "goal", "id": "g1", "goal": "...", "check": ..., "budget": 8}}. check makes
the goal close by itself when it holds: {{"name": "holding", "hand": "left|right"}} (that hand holds the object its
last pick verified), {{"name": "hand_free", "hand": ...}}, {{"name": "placed"}} (a place call in this goal returned
OK) or {{"name": "arrived"}} (a navigate call in this goal returned OK); null when you will close the goal yourself.
budget is the most skill calls the goal may take (1 to 15). At most 12 goals.

To write or rewrite the plan, reply with skill plan and arguments {{"tree": <the tree>}}. While a goal is active,
call skills for it; when it is reached and has no check, call goal_done with a short reason; when it cannot be
reached, call goal_failed with the reason, and the tree moves to its next alternative. If the tree fails, you may
rewrite the rest of it at most {plan.replans} times; goals already done stay done. When the tree is done, or
cannot go on, finish with done.
