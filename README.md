# cs462searchalgos

Hello, this is my searching algorithms programs for CS462. In this program I implement the following search algorithms:
  undirected (blind) brute-force approach 
  breadth-first search
  depth-first search
  ID-DFS search
  best-first search
  A* search

This uses the latest version of python and some libraries may have to be installed such as matplotlib

I also implement a way graph each individual algorithm or to run all of them and compare the results. Many parts of the 
search algorithms themselves were built using a generative model as well as graphing algorithm, below I describe how
I used the model to create this. I also had to add the city 'Hays' to coordinates.csv because it was missing.

To create this program with the help of a generative model I first built all of the file input and built my initial
coords and adjecncies dictionaries and laid out a simple user-io system. I started working on my own brute force implementation
and found the parameters needed: start, end, coords, adj. Using these as a baseline I ended up prompting the model to
create the other implementations using these params and I provided what my data structures looked like so I would
be sure it used them correctly. After that and much tweaking / debugging I had algorithms that appeared to work, but with no
way to validate them. So after that I started creating the way I wanted to graph the path, I chose to use matplotlib and asked
the model to create a way for me to graph it. This was actually most of the challenge in this project as there was a lot of errors
and tweaking needed to get the end result wanted.

I hope you enjoy my program!
