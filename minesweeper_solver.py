import time
import pygame
import os
import random
from sympy import And, Not, Or, symbols
from sympy.logic.inference import satisfiable
from itertools import combinations


class Game:
    def __init__(self, board, screenSize):
        self.board = board
        self.screenSize = screenSize
        self.solver = board.solver
        self.pieceSize = (
            self.screenSize[0] // self.board.getSize()[1],
            self.screenSize[1] // self.board.getSize()[0],
        )
        self.loadImages()

    def run(self, run_with_solver):
        """
        return: The returns a tuple of two values. The first value is a boolean value that is
        True if the game was won and False if the game was lost. The second value is the time it took to
        win or lose the game.
        """
        pos_x = 1920 / 2 - self.screenSize[0] / 2
        pos_y = 1080 / 2 - self.screenSize[1] / 2
        os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (pos_x, pos_y)
        pygame.init()
        self.screen = pygame.display.set_mode(self.screenSize)
        running = True
        while running:
            if run_with_solver:
                piece, rightClick = self.solver.nextClick(self.board)
                self.board.handleClick(piece, rightClick)

            # Checking if the mouse is clicked and if it is, it is getting the position of the mouse
            # and checking if it is a right click.
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        position = pygame.mouse.get_pos()
                        rightClick = pygame.mouse.get_pressed()[2]
                        piece = self.handleClick(position, rightClick)
            
            self.draw()
            pygame.display.flip()
            if self.board.getWon() or self.board.getLost():
                running = False
                time.sleep(0.1)
        pygame.quit()
        return self.board.getWon(), time.time()

    def draw(self):
        """
        For each row and column in the board, get the piece at that location, get the image for that
        piece, and draw the image at the top left corner of the screen
        """
        topLeft = (0, 0)
        for row in range(self.board.getSize()[0]):
            for col in range(self.board.getSize()[1]):
                piece = self.board.getPiece((row, col))
                image = self.getImage(piece)
                self.screen.blit(image, topLeft)
                topLeft = topLeft[0] + self.pieceSize[0], topLeft[1]
            topLeft = 0, topLeft[1] + self.pieceSize[1]

    def loadImages(self):
        """
        It loads all the images in the images folder and scales them to the size of the pieces
        """
        self.images = {}
        for fileName in os.listdir("images"):
            if not fileName.endswith(".png"):
                continue
            image = pygame.image.load(r"images/" + fileName)
            image = pygame.transform.scale(image, self.pieceSize)
            self.images[fileName.split(".")[0]] = image

    def getImage(self, piece):
        """
        If the piece is clicked, return the image of the bomb or the number of bombs around it. If the
        piece is not clicked, return the image of the flag or the empty block
        
        piece: the piece that is being drawn
        return: The image of the piece.
        """
        string = None
        if piece.getClicked():
            string = (
                "bomb-at-clicked-block"
                if piece.getHasBomb()
                else str(piece.getNumAround())
            )
        else:
            string = "flag" if piece.getFlagged() else "empty-block"
        return self.images[string]

    def handleClick(self, position, rightClick):
        """
        If the game is not lost, get the piece at the position of the click and handle the click
        
        position: The position of the mouse click
        rightClick: Boolean
        return: The return value is the value of the last expression evaluated.
        """
        if self.board.getLost():
            return
        index = position[1] // self.pieceSize[1], position[0] // self.pieceSize[1]
        piece = self.board.getPiece(index)
        self.board.handleClick(piece, rightClick)


class Board:
    def __init__(self, size, numBombs, prob = 0.02):
        self.size = size
        self.prob = prob
        self.lost = False
        self.numClicked = 0
        self.settedBombs = 0
        self.numBombs = numBombs
        self.solver = Solver(self.size)
        self.createBoard()
        self.getNumNonBombs()
        self.setBoard()

    def createBoard(self):
        """
        It creates a board of size self.size[0] by self.size[1] and fills it with Piece objects
        """
        self.board = []
        for i in range(self.size[0]):
            row = []
            for j in range(self.size[1]):
                row.append(Piece(False, (i, j)))
            self.board.append(row)

    def setBoard(self):
        """
        It sets the board by randomly placing bombs on the board until the number of bombs placed is
        equal to the number of bombs specified by the user
        """

        while self.settedBombs < self.numBombs:
            for row in range(self.size[0]):
                for col in range(self.size[1]):
                    if self.settedBombs == self.numBombs:
                        break
                    if self.board[row][col].getHasBomb():
                        continue
                    hasBomb = random.random() < self.prob
                    if hasBomb:
                        self.settedBombs += 1
                        self.board[row][col] = Piece(hasBomb, (row, col))
        self.setNeighbors()

    def setNeighbors(self):
        """
        For each piece in the board, set the list of neighbors for that piece to be the list of
        neighbors for that piece
        """
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                piece = self.getPiece((row, col))
                neighbors = self.getListOfNeighbors((row, col))
                piece.setNeighbors(neighbors)

    def getListOfNeighbors(self, index):
        """
        It returns a list of all the pieces that are adjacent to the piece at the given index
        
        index: the index of the piece you want to get the neighbors of
        return: A list of the neighbors of the piece at the given index.
        """
        neighbors = []
        for row in range(max(0, index[0] - 1), min(index[0] + 2, self.size[0])):
            for col in range(max(0, index[1] - 1), min(index[1] + 2, self.size[1])):

                if row == index[0] and col == index[1]:
                    continue
                neighbors.append(self.getPiece((row, col)))
        return neighbors

    def getSize(self):
        """
        It returns the size of the board.
        return: The size of the board
        """
        return self.size

    def getPiece(self, index):
        """
        It returns the piece at the given index.
        
        index: a tuple of the form (row, column)
        return: The piece at the given index.
        """
        return self.board[index[0]][index[1]]

    def handleClick(self, piece, flag):
        """
        If the piece is clicked or flagged. If the piece is flagged, toggle the flag.
        Click the piece and if it has a bomb, set lost to true. If the piece has no
        bomb, add 1 to the number of clicked tiles. If the tile has no bombs around then
        recursevely click on all his neighbors

        piece: the piece that was clicked
        flag: True if the user right-clicked, False if the user left-clicked
        return: The return statement is used to exit a function and go back to the place from where it
        was called.
        """
        if piece.getClicked() or (not flag and piece.getFlagged()):
            return
        if flag:
            piece.toggleFlag()
            self.settedBombs -= 1
            return
        piece.click()
        if piece.getHasBomb():
            self.lost = True
            return
        self.numClicked += 1
        if piece.getNumAround() != 0:
            self.solver.toCheck.append(piece)
            return
        for neighbor in piece.getListOfNeighbors():
            if not neighbor.getHasBomb() and not neighbor.getClicked():
                self.handleClick(neighbor, False)

    def getWon(self):
        """
        If the number of non-bombs that have been clicked is equal to the number of non-bombs in the
        game, then the game has been won
        return: The number of non-bombs that have been clicked.
        """
        return self.numNonBombs == self.numClicked

    def getLost(self):
        """
        Getter for the self.lost value
        return: The lost variable is being returned.
        """
        return self.lost

    def getNumNonBombs(self):
        """
        This function calculates the number of non-bombs in the game by subtracting the number of bombs
        from the total number of tiles
        """
        self.numNonBombs = self.size[0] * self.size[1] - self.numBombs

    def getNonClickedNonFlagged(self):
        """
        It returns a list of all the pieces on the board that are not clicked and not flagged
        return: A list of all the pieces that are not clicked or flagged.
        """
        toRet = []
        for row in self.board:
            for piece in row:
                if not (piece.clicked or piece.getFlagged()):
                    toRet.append(piece)
        return toRet

    def getNeighborsNonClickedNonFlagged(self, toCheck):
        neighbors = set()
        for piece in toCheck:
            for neighbor in piece.getListOfNeighbors():
                if not (neighbor.getClicked() or neighbor.getFlagged()):
                    neighbors.add(neighbor)
        return neighbors


# A Piece is a tile of the board.
class Piece:
    def __init__(self, hasBomb, coordinates=None):
        """
        The function __init__() is a constructor that initializes the variables hasBomb, clicked,
        flagged, and coordinates.
        
        hasBomb: Boolean, True if the tile has a bomb, False if it doesn't
        coordinates: The coordinates of the tile
        """

        self.hasBomb = hasBomb
        self.clicked = False
        self.flagged = False
        self.coordinates = coordinates
        self.PL_variable = symbols(
            str(self.coordinates[0]) + "_" + str(self.coordinates[1])
        )

    def getHasBomb(self):
        """
        It returns the value of the hasBomb variable.
        return: The hasBomb variable is being returned.
        """
        return self.hasBomb

    def getClicked(self):
        """
        It returns the value of the variable "clicked"
        return: The value of the clicked variable.
        """
        return self.clicked

    def getFlagged(self):
        """
        This function returns the value of the flagged variable
        return: The value of the flagged variable.
        """
        return self.flagged

    def getNumAround(self):
        """
        This function returns the number of mines around a tile
        return: The number of mines around the tile.
        """
        return self.numAround

    def getNumFlaggedAround(self):
        """
        It returns the number of flagged neighbors of a cell
        return: The number of flagged neighbors.
        """
        num = 0
        for neighbor in self.neighbors:
            num += 1 if neighbor.getFlagged() else 0
        return num

    def getNumUnclickedAround(self):
        """
        For every neighbor cell if the cell has been clicked, add 0, otherwise add 1
        return: The number of unclicked neighbors.
        """
        num = 0
        for neighbor in self.neighbors:
            num += 0 if neighbor.getClicked() else 1
        return num

    def getUnclickedAroundUnflagged(self):
        """
        It returns a list of all the neighbors of the current cell that are not clicked and not flagged
        return: A list of neighbors that are not clicked or flagged.
        """
        return [
            neighbor
            for neighbor in self.neighbors
            if not (neighbor.getClicked() or neighbor.getFlagged())
        ]

    def hasClickedNeighbor(self):
        """
        If any of the neighbors of the current cell have been clicked, return True. Otherwise, return
        False
        return: a boolean value.
        """
        for neighbor in self.neighbors:
            if neighbor.getClicked():
                return True
        return False

    def setNeighbors(self, neighbors):
        """
        This function takes in a list of neighbors and sets the neighbors of the current node to the
        list of neighbors
        
        neighbors: a list of the neighbors of the cell
        """
        self.neighbors = neighbors
        self.setNumAround()

    def setNumAround(self):
        """
        This function sets the number of bombs around a piece to 0, then iterates through the neighbors
        of the piece and adds 1 to the number of bombs around the piece for each neighbor that has a
        bomb
        """
        self.numAround = 0
        for piece in self.neighbors:
            if piece.getHasBomb():
                self.numAround += 1

    def getListOfNeighbors(self):
        """
        It returns the neighbors of the current node.
        return: The list of neighbors
        """
        return self.neighbors

    def toggleFlag(self):
        """
        It takes a cell object and toggles the flagged attribute of that cell object
        """
        self.flagged = not self.flagged

    def click(self):
        """
        Setter of the clicked attribute of the object that called the function to True
        """
        self.clicked = True

    def neighborsOfNeighbors(self):

        result = set()

        for neighbor in self.getListOfNeighbors():
            for externalNeighbor in neighbor.getListOfNeighbors():
                if not (externalNeighbor.getClicked() or externalNeighbor.getFlagged()):
                    result.add(externalNeighbor)

        return result



class Solver:
    def __init__(self, size):
        self.queue = set()
        self.toCheck = []
        self.size = size
        self.corners = [
            (0, self.size[1] - 1),
            (self.size[0] - 1, 0),
            (self.size[0] - 1, self.size[1] - 1),
            (0, 0),
        ]

    def nextClick(self, board):
        """
        If there are any pieces in the queue, return the last one. Otherwise, 
        if there are any pieces in the corners list, return the last one
        
        board: the board object
        return: The piece and a boolean value.
        """

        if self.queue:
            return self.queue.pop()

        self.toCheck = [
            piece
            for piece in self.toCheck
            if piece.getClicked()
            and piece.getNumUnclickedAround() - piece.getNumFlaggedAround() > 0
        ]

        self.basic()

        if self.queue:
            return self.queue.pop()

        self.csp(board)
        # self.advanced()

        if self.queue:
            return self.queue.pop()

        if self.corners:
            return board.getPiece(self.corners.pop()), False

        return board.getPiece((0, 0)), False

    def basic(self):
        """
        If the number of flags around a piece is equal to the number of mines around it, then all the
        unclicked pieces around it are safe. If the number of unclicked pieces around it is equal to the
        number of mines around it, then all the unclicked pieces around it are mines.
        """
        for piece in self.toCheck:
            if piece.getClicked():
                notClickedNotFlagged = []
                flagAround = 0
                for neighbor in piece.getListOfNeighbors():
                    if not neighbor.getClicked() and not neighbor.getFlagged():
                        notClickedNotFlagged.append(neighbor)
                    else:
                        if neighbor.getFlagged():
                            flagAround += 1

                if len(notClickedNotFlagged) + flagAround == piece.numAround:
                    self.queue = self.queue.union(
                        {(x, True) for x in notClickedNotFlagged}
                    )

                elif flagAround == piece.numAround:
                    self.queue = self.queue.union(
                        {(x, False) for x in notClickedNotFlagged}
                    )

    def advanced(self):

        listOfSegregated, listOfSegregatedNeighbors = self.segregating()

        for i in range(len(listOfSegregated)):
            formula = set()
            for piece in listOfSegregated[i]:

                valid = set(
                    [piece.PL_variable for piece in piece.getUnclickedAroundUnflagged()]
                )
                valueToAim = piece.getNumAround() - piece.getNumFlaggedAround()

                combs = [set(x) for x in combinations(valid, valueToAim)]
                pieceFormula = []
                for comb in combs:
                    pieceFormula.append(
                        And(*comb.union(Not(piece) for piece in valid.difference(comb)))
                    )
                pieceFormula = Or(*pieceFormula)
                formula.add(pieceFormula)
            formula = And(*formula)

            print(formula)
            for neighbor in listOfSegregatedNeighbors[i]:

                val = neighbor.PL_variable
                print(val)
                if satisfiable(And(formula, val), algorithm="dpll2") == False:

                    self.queue.add((neighbor, False))

                if satisfiable(And(formula, Not(val)), algorithm="dpll2") == False:

                    self.queue.add((neighbor, True))

    def csp(self, board):
        """
        It takes the board, segregates the board into groups of cells that are connected to each other,
        and then checks the constraints of each group
        
        board: the board object
        """

        listOfSegregated, listOfSegregatedNeighbors = self.segregating()
        variables = board.getNonClickedNonFlagged()
        probs = {}

        if len(variables) < 10:
            probs.update(self.countConstraint(variables))

        for segregated, neighbors in zip(listOfSegregated, listOfSegregatedNeighbors):
            probs.update(self.neighborsConstraint(segregated, neighbors))

        if probs:
            self.checkSecurePieces(probs)

    def segregating(self):
        """
        It returns a list of lists of pieces that are connected to each other by a common piece
        return: A list of lists of pieces that are connected to each other.
        """

        segregated = self.segregatingClickedPieces()
        neighborsOfSegregated = self.neighborsOfSegregatedPieces(segregated)

        return self.connectWithCommonPiece(segregated, neighborsOfSegregated)

    def segregatingClickedPieces(self):
        """
        It takes a list of pieces, and returns a list of sets of pieces, where each set is a group of
        pieces that are connected to each other
        return: A list of sets of pieces.
        """
        segregated = []
        toCheckDestroyed = self.toCheck.copy()
        # Segregating the pieces into groups of connected pieces.
        while toCheckDestroyed:
            temp = [toCheckDestroyed.pop()]
            for piece in temp:
                for neighbor in piece.getListOfNeighbors():
                    if neighbor in toCheckDestroyed:
                        temp.append(neighbor)
                        toCheckDestroyed.remove(neighbor)
            segregated.append(set(temp))
        return segregated

    def neighborsOfSegregatedPieces(self, segregated):
        """
        It takes a list of segregated pieces and returns a list of sets of neighbors of those segregated
        pieces
        
        segregated: a list of lists of pieces. Each list of pieces is a segregated group
        return: A list of sets of pieces.
        """
        neighborsOfSegregated = []
        # Finding the neighbors of the segregated pieces.
        for segr in segregated:
            neighbors = set()
            for piece in segr:
                for neighbor in piece.getUnclickedAroundUnflagged():
                    neighbors.add(neighbor)

            neighborsOfSegregated.append(neighbors)
        return neighborsOfSegregated

    def connectWithCommonPiece(self, segregated, neighborsOfSegregated):
        """
        It takes a list of sets of nodes and a list of sets of neighbors of those nodes, and returns a
        new list of sets of nodes and a new list of sets of neighbors of those nodes, where two sets
        of nodes are unified, both the nodes and the neighbors, if they have at least one common neighbor
        
        segregated: list of lists of nodes that are segregated
        neighborsOfSegregated: a list of sets of neighbors of each segregated piece
        return: The return value is a tuple of two lists. The first list contains the segregated nodes,
        and the second list contains the neighbors of the segregated nodes.
        """

        toRetSegragated = []
        toRetNeighbors = []

        # Finding the connected components of a graph.
        while len(neighborsOfSegregated) > 0:
            first, *rest = neighborsOfSegregated
            firstSegr, *restSegr = segregated
            lf = -1
            while len(first) > lf:
                lf = len(first)
                rest2 = []
                restSegr2 = []
                for r, rSegr in zip(rest, restSegr):
                    if len(first.intersection(r)) > 0:
                        first |= r
                        firstSegr |= rSegr
                    else:
                        rest2.append(r)
                        restSegr2.append(rSegr)

                rest = rest2
                restSegr = restSegr2
            toRetNeighbors.append(list(first))
            toRetSegragated.append(list(firstSegr))
            neighborsOfSegregated = rest
            segregated = restSegr
        return toRetSegragated, toRetNeighbors

    def checkSecurePieces(self, probs):
        """
        If there are no pieces that are 100% or 0% of being a bomb, then call the function that picks
        the piece with the closest probability of being, or not being, a bomb
        
        probs: a dictionary of pieces and their probabilities
        """
        flag = True
        for piece in probs:
            if probs[piece] > 0.99:
                self.queue.add((piece, True))
                flag = False
            elif probs[piece] < 0.01:
                self.queue.add((piece, False))
                flag = False

        if flag and not self.corners:
            self.chooseWithLowerProb(probs)

    def countConstraint(self, variables):
        """
        It takes a list of variables, creates a CSP with those variables and a domain of 0 and 1, adds a
        constraint that the sum of the variables must be equal to the number of bombs on the board, and
        then returns the probability of each variable being a bomb
        
        variables: a list of the unclicked pieces on the board
        return: A dictionary of the probabilities of each piece being a bomb.
        """
        csp = CSP(variables, {piece: [0, 1] for piece in variables})
        constraint = Constraint(variables, board.settedBombs)
        csp.add_constraint(constraint)

        csp.backtracking_search()
        solutions = csp.solutions
        return self.probabilities(solutions, variables)

    def neighborsConstraint(self, segregated, neighbors):
        """
        It takes a list of segregated pieces and the list their neighbors, and returns a dictionary of the
        probabilities of each neighbor being a mine
        
        segregated: a list of pieces that still have at least one piece not clicked
        that are segregated from the rest of valid pieces to reduce the exponential time
        neighbors: the list of the unclicked pieces that are neighbors of the segregated pieces
        return: A dictionary of the probabilities of each neighbor being a mine.
        """
        csp = CSP(
            segregated + neighbors,
            {piece: [0, 1] for piece in neighbors}
            | {piece: [0] for piece in segregated},
        )

        for piece in segregated:
            constraint = Constraint(
                piece.getUnclickedAroundUnflagged(),
                piece.numAround - piece.getNumFlaggedAround(),
            )
            csp.add_constraint(constraint)

        csp.backtracking_search()
        solutions = csp.solutions
        return self.probabilities(solutions, neighbors)

    def chooseWithLowerProb(self, probs):
        """
        If the minimum probability is less than or equal to 1 minus the maximum probability, then add
        the minimum probability to the queue with a False value. Otherwise, add the maximum probability
        to the queue with a True value (If True than a flag will be placed in that square, if False then 
        that square will be left clicked)
        
        probs: a dictionary of probabilities for each item in the queue
        """

        minimum = min(probs, key=probs.get)
        maximum = max(probs, key=probs.get)

        if probs[minimum] <= 1 - probs[maximum]:
            self.queue.add((minimum, False))
        else:
            self.queue.add((maximum, True))

    def probabilities(self, solutions, neighbors):
        """
        For each neighbor, we add the number of times it appears in each solution divided by the total
        number of solutions
        
        solutions: a list of dictionaries, each dictionary is a solution to the problem
        neighbors: a list of all the neighbors of the current node
        return: The probabilities of each piece being in the solution.
        """

        probs = {neighbor: 0 for neighbor in neighbors}
        totalSolutions = len(solutions)

        for solution in solutions:
            for piece in solution:
                if piece in probs:
                    probs[piece] += solution[piece] / totalSolutions
        return probs


# Constraint is a class that represents a constraint on a variable.
class Constraint:
    def __init__(self, variables, value):
        self.variables = variables
        self.value = value

    def satisfied(self, assignment):
        """
        If all the variables in the constraint are assigned, then check if the sum of the assigned
        values is equal to the value of the constraint. Otherwise, check if the sum of the assigned
        values is less than or equal to the value of the constraint
        
        assignment: a dictionary of the form {'A': 1, 'B': 0, ...} where the keys are variables
        and the values are the assigned values for those variables
        """

        count = 0
        flag = True
        for piece in self.variables:
            if piece in assignment:
                count += assignment[piece]
            else:
                flag = False
        if flag:
            return count == self.value
        return count <= self.value


# The CSP class is a data structure that stores the variables, domains, and constraints of a CSP problem.
class CSP:
    def __init__(self, variables, domains):
        self.variables = variables
        self.domains = domains
        self.constraints = {}
        for variable in self.variables:
            self.constraints[variable] = []
        self.solutions = []

    def add_constraint(self, constraint):
        """
        It adds a constraint to the constraints dictionary.
        
        constraint: a Constraint object
        """
        for variable in constraint.variables:
            self.constraints[variable].append(constraint)

    def consistent(self, variable, assignment):
        """
        If all the constraints for a variable are satisfied, then the assignment is consistent
        
        variable: The variable we're currently assigning
        assignment: a dictionary of variable:value pairs
        return: a boolean value.
        """
        for constraint in self.constraints[variable]:
            if not constraint.satisfied(assignment):
                return False
        return True

    def backtracking_search(self, assignment={}):
        """
        If the assignment is complete, add it to the list of solutions. Otherwise, for each value in the
        domain of the first unassigned variable, add it to the assignment and recursively call the function
        
        :param assignment: a dictionary of the current assignment
        """

        if len(assignment) == len(self.variables):
            self.solutions.append(assignment)

        else:
            unassigned = [v for v in self.variables if v not in assignment]

            first = unassigned[0]

            for value in self.domains[first]:
                local_assignment = assignment.copy()
                local_assignment[first] = value

                if self.consistent(first, local_assignment):
                    self.backtracking_search(local_assignment)


if __name__=="__main__":
    size = (16, 32)
    numBombs = 60
    screenSize = (size[1] * 30, size[0] * 30)
    count = 0



    board = Board(size, numBombs)
    game = Game(board, screenSize)
    start = time.time()
    temp, end = game.run(run_with_solver=False)
    

