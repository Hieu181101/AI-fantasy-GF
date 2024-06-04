# AI-Fantasy-GF
So yah being a CS student meaning that I do not have the time to touch grass and after watching the move "Her" I really want an AI GirlFriend. So I decide why not make one ? Who needs a real Girlfriend anyway

## The Goals:
- She should be able to run with a decent GPU
- She can understand me well enough where if I ask about what is the distance to the moon she would not talk about sushi
- I can text to her and speak to her
- She can text back and speak to me
- I can see my GF
- I can change her apperance

## Communication is KEY (in any relationship) 
First she kinda needs a brain where she can talk to me. This step is quite easy with all the AI model like ChatGPT or Gemini; we can just use those for our GF brain. I will use the model 3 turbo from Open AI since I want my GF to be super smart but feel free to change. 
And adding my perfect imagination of what my GF would behave to the Prompt and Walla she can now speak to me. (just through text only first)
One more thing: She kinda needs to remeber what we are talking about so I implement a History holder that would retain the conversation we have 

Let's move on to how she can hear me and be able to speak back: This part is easy enough; we can use TTS and STT for our communication. 



<div>
  <img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB" alt="react icon">
  </br>
  <img src="https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E" alt="javascript icon">
  <br>
  <img src="https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white" alt="typescript icon">
  </br>
  <img src="https://img.shields.io/badge/npm-CB3837.svg?style=for-the-badge&logo=npm&logoColor=white" alt="npm">
</div>

## Maze Generation 

- **Recursive Backtracking**: First choose a starting point in the field (for our case is always the first block), than randomly choose a wall at that point and carve a passage through to the adjacent block, but only if the adjacent block has not been visited yet. If all adjacent blocks have been visited, back up to the last block that has uncarved walls and repeat. The algorithm ends when the process has backed all the way up to the starting point.
- **Binary Tree**: First choose a starting point in the field (for our case is always the first block). For each block, randomly carve a passage either to the north or to the east, but only if the adjacent block in that direction exists. If neither direction is available, move to the next block in the grid. The algorithm continues until all blocks have been processed. For the case of Binary Tree we can switch it up with north/east, south/west, or south/east but I decide to use North and East.
- **Prim**: Prim's algorithm begins by choosing a random starting block. It then adds all walls of the initial block to a list. The algorithm randomly selects a wall from the list and carves a passage through to the adjacent block, but only if the adjacent block has not been visited yet. This new block's walls are then added to the list. If a wall leads to an already visited block, it is discarded. The process repeats until all blocks have been visited, resulting in a fully generated maze.
- **Kruskal**: Kruskal's algorithm starts by initializing each block as its own set. It then randomly simply select an edge at random, and join the blocks it connects if they are not already connected by a path. The algorithm continues selecting and removing walls until all blocks are connected, ensuring that no cycles are formed. This process results in a maze where each block is reachable from any other block.

**Reference**: The Buckblog assorted ramblings by Jamis Buck [Click here](https://weblog.jamisbuck.org/2010/12/27/maze-generation-recursive-backtracking)

## Search Algorithm 

- **BFS**: BFS algorithm starts at the start location and explores all its neighboring blocks before moving on to their neighbors. It uses a queue to keep track of blocks to be explored. BFS would search all the neighbors for the goal before moving on. The algorithm ends when the goal block is reached or all possible paths have been explored.

- **DFS**: DFS algorithm starts at the start location and explores as far as possible along each branch before backtracking. It uses a stack to keep track of blocks to be explored, and keep going at one path until it can go no further before backtracking to the most recent block and exploring alternative paths. The algorithm ends when the goal block is reached or all possible paths have been explored.

- **Dijkstra**: Dijkstra's algorithm starts at the start location and explores the maze by expanding the shortest known path to each neighboring block to the goal. Using a priority queue to always expand the least costly path first. Each block is assigned a distance value, representing the shortest known distance from the start block, which is updated as shorter paths are found. The algorithm ends when the goal block is reached with the shortest possible path.

- **A***: The A* Algorithm starts at the designated start location and uses both the actual cost from the start (g(x)) and a heuristic estimate of the cost to the goal (h(x)) to determine the shortest path to the goal. It maintains a priority queue to expand the cell with the lowest total cost of (f(x) = g(x) + h(x)). The heuristic function would estimates the shortest path to the goal, guiding the search more efficiently towards the target. The algorithm ends when the goal cell is reached with the most cost-effective path. 

## Look Through
![Screenshot (3)](https://github.com/Hieu181101/MAZA-visulization/assets/135567323/bcac7ded-c160-42b3-bc8e-e3994f4de9d5)

![Screenshot (10)](https://github.com/Hieu181101/MAZA-visulization/assets/135567323/af1d346e-8d36-42ba-9e7f-d909bb1bc936)

![Screenshot (6)](https://github.com/Hieu181101/MAZA-visulization/assets/135567323/1c7e18b0-43a4-43f0-b3fb-2a0ce732a1aa)
## Getting Started 
To get a local copy up and running follow these simple steps.
### Prerequisites

Make sure you have Node.js and npm installed. You can download them from [Node.js](https://nodejs.org/).

## Steps: 

1. Clone the repository (HTTPS).
  ```sh
  git clone https://github.com/Hieu181101/MAZA-visulization
  ```
2. Navigate to directory
  ```sh
  cd MAZA-visulization
  ```
2. Install NPM packages
  ```sh
  npm install
  ```
3. Run the project
  ```sh
  npm start
  ```



