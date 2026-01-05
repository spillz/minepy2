# Project: minepy2 - a minecraftlike in pure python

## General Instructions

- This is a Minecraft-like game written in Python, Pyglet and Numpy.
- Ignore the web folder unless told explicitly to look at JavaScript (an incomplete port of the game).
- The program use a mix of OO and imperative style, avoid functional programming techniques.
- Prefer vectorized operations with numpy over loops when working on numerically heavy codepaths.
- Speedy rendering and calculation performance are crucial.
- Do NOT attempt to run `python main.py` as that can cause issues. Instead instruct the user to test changes if you want feedback or add a standalone test.
- If tests are ever needed, put them in the tests folder. Keep them to a minimum.
- Rather than throwing away code on large speculative changes, prefer to add a separate function/class/module to test those ideas with a code toggle to flip between old and new behavior. 
- If you are uncertain about how to proceed, prompt the user for more information.
