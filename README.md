# Minesweeper AI Solver 🧠💣

This project is an Artificial Intelligence (AI) algorithm that plays the classic **Minesweeper** game. The game is implemented using **Pygame** and the AI uses a **Constraint Satisfaction Problem (CSP)** solver to efficiently solve the game. This solver can handle a variety of board sizes and mine configurations.

## Features ✨
- **Minesweeper Implementation**: A playable version of the game is built using the Pygame library.
- **AI Solver**: The AI solves the game using a CSP-based algorithm.
- **Customizable Game Settings**: You can configure the board size and number of bombs.

## Gameplay 🎮

The game involves a grid of cells, with some cells containing hidden bombs. The goal is to uncover all safe cells without detonating any bombs. The AI algorithm automatically detects where it is safe to click based on surrounding clues, solving the game as efficiently as possible.

## How It Works 🧩

The AI leverages **Constraint Satisfaction Problem (CSP)** techniques to deduce which cells contain bombs based on the numbers revealed around a clicked cell. The algorithm makes logical deductions to either flag mines or safely reveal more cells.

## Technical Details 📝

For more detailed technical explanations of how the **CSP solver** is implemented, please refer to the `minesweeper_solver.pdf` document included in this repository. It contains an in-depth discussion of the constraint satisfaction model, logical inference techniques, and optimizations used to improve solver efficiency.

## How to Run the Game with the Solver 🕹️

### 1. **Install Dependencies**:
   - First, make sure you have **Python 3.x** installed.
     
  The project relies on the following libraries:
   - `pygame`: For the graphical interface and game mechanics.
   - `sympy`: For logical inference and building the CSP solver.
     
  Install the required libraries by running:
     ```bash
     pip install pygame sympy
     ```
### 2. **run the game**:
  - Once the dependencies are installed, you can run the game and the solver with `main.py`
  - if you want to run only the game without the solver run `mindsweeper_solver.py`

## Configuring Game Settings ⚙️

You can easily customize the game settings by modifying the parameters in the main script. Adjust the following variables to change the board size and the number of bombs:

 - Board Size: Modify the grid dimensions.
 - Bomb Count: Set the number of bombs hidden in the grid.
