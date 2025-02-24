import pygame
import math
from queue import PriorityQueue
import numpy as np
from pygame import mixer

WIDTH = 800
HEIGHT = 900
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Path Finding Algorithm")

# pre-defining-colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 128, 0)
GRAY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

#main blueprint of box
class Cubes:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = YELLOW

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

#h-function; here we have used manhattan function to calculate the heuristic distance
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2 #calculatin
    return abs(x1 - x2) + abs(y1 - y2)

#for the beep sound
#we have used python mixer for this
def play_search_beep(distance, max_distance):
    if not mixer.get_init():
        mixer.init(frequency=44100, size=-16, channels=1)

    frequency = 200 + (800 * (1 - (distance / max_distance)))
    sample_rate = 44100
    duration = 0.05
    samples = int(sample_rate * duration)

    t = np.linspace(0, duration, samples, False)
    audio = np.sin(2 * np.pi * frequency * t)
    fade_samples = samples // 4 #fading so that we dont get that buzzing sound
    envelope = np.concatenate((
        np.linspace(0, 1, fade_samples) ** 2,
        np.ones(samples - 2 * fade_samples),
        np.linspace(1, 0, fade_samples) ** 2
    ))
    audio *= envelope
    sound_array = (audio * 32767).astype(np.int16)

    sound = mixer.Sound(sound_array)
    sound.play()


def play_path_beep():
    if not mixer.get_init():
        mixer.init(frequency=44100, size=-16, channels=1)

    frequency = 600
    sample_rate = 44100
    duration = 0.03
    samples = int(sample_rate * duration)

    t = np.linspace(0, duration, samples, False)
    audio = 2 * np.abs((frequency * t) % 1 - 0.5) - 0.5
    fade_samples = samples // 4
    envelope = np.concatenate((
        np.linspace(0, 1, fade_samples) ** 2,
        np.ones(samples - 2 * fade_samples),
        np.linspace(1, 0, fade_samples) ** 2
    ))
    audio *= envelope
    sound_array = (audio * 32767).astype(np.int16)

    sound = mixer.Sound(sound_array)
    sound.play()


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        play_path_beep()
        draw()
        pygame.time.wait(20) #delay


def main_algo(draw, grid, start, end):
    count = 0
    open_queue = PriorityQueue()
    open_queue.put((0, count, start)) # we got count to compare if two of the weights are of same value then we will use the one which came first to the queue
    came_from = {}
    g_score = {cube: float("inf") for row in grid for cube in row}
    g_score[start] = 0
    f_score = {cube: float("inf") for row in grid for cube in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    max_distance = h((0, 0), (grid[0][0].total_rows - 1, grid[0][0].total_rows - 1))

    open_queue_hash = {start}

    while not open_queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_queue.get()[2]
        open_queue_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            return True

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + h(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_queue_hash:
                    count += 1
                    open_queue.put((f_score[neighbour], count, neighbour))
                    open_queue_hash.add(neighbour)
                    neighbour.make_open()
                    current_distance = h(neighbour.get_pos(), end.get_pos())
                    play_search_beep(current_distance, max_distance)

        draw()

        if current != start:
            current.make_closed()

    return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cube = Cubes(i, j, gap, rows)
            grid[i].append(cube)
    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GRAY, (i * gap, 0), (i * gap, width))
        for j in range(rows):
            pygame.draw.line(win, GRAY, (0, j * gap), (width, j * gap))


def draw_controls(win):
    font = pygame.font.SysFont('arial', 16)  # Reduced font size from 20 to 16
    controls = [
        "Controls:",
        "Left Click: Set Start (Orange), End (Turquoise), or Barrier (Black)",
        "Right Click: Remove Start/End/Barrier",
        "Space: Start A* Algorithm",
        "C: Clear Grid"
    ] #for the ui part

    for i, text in enumerate(controls):
        label = font.render(text, 1, BLACK)
        win.blit(label, (10, 805 + i * 18))


def draw(win, grid, rows, width, show_controls):
    win.fill(WHITE)
    for row in grid:
        for cube in row:
            cube.draw(win)
    draw_grid(win, rows, width)
    if show_controls:
        draw_controls(win)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    if 0 <= x < width and 0 <= y < width:
        row = y // gap # = integer division
        col = x // gap
        return row, col
    return None


def main(win, width):
    pygame.font.init()
    ROWS = 50
    grid = make_grid(ROWS, width)
    start = None
    end = None
    run = True
    show_controls = True

    while run:
        draw(win, grid, ROWS, width, show_controls)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                clicked = get_clicked_pos(pos, ROWS, width)
                if clicked:
                    row, col = clicked
                    cube = grid[row][col]
                    if not start and cube != end:
                        start = cube
                        start.make_start()
                    elif not end and cube != start:
                        end = cube
                        end.make_end()
                    elif cube != end and cube != start:
                        cube.make_barrier()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                clicked = get_clicked_pos(pos, ROWS, width)
                if clicked:
                    row, col = clicked
                    cube = grid[row][col]
                    cube.reset()
                    if cube == start:
                        start = None
                    elif cube == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    show_controls = False
                    for row in grid:
                        for cube in row:
                            cube.update_neighbours(grid)
                    main_algo(lambda: draw(win, grid, ROWS, width, False), grid, start, end)
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                    show_controls = True #showing controls again after clearing the screen

    pygame.quit()


main(WIN, WIDTH)