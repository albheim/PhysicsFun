import numpy as np
import pygame

width, height = 800, 600
fps = 60
dt = 1.0 / fps * 10

k = 100
k_mouse = 10000
length = 10
m = 10
g = np.array([0, 9.82])

n_nodes = 20
p = np.zeros((n_nodes, 2))
pdot = np.zeros((n_nodes, 2))
pdotdot = np.zeros((n_nodes, 2))
F = np.zeros(n_nodes)

p[:, 0] = width / 2
p[:, 1] = height / 2

p[:, 0] += np.linspace(0, width / 2.5, n_nodes)

pygame.init()
screen = pygame.display.set_mode((width, height))
done = False
clock = pygame.time.Clock()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    mp = np.array(pygame.mouse.get_pos()).reshape((1, 2))
    # simulate mouse sitting still to the middle right of screen
    # mp = np.array([[0.9 * width, height / 2]])

    dp = p - np.append(p[1:], mp, axis=0)
    pdist = np.maximum(np.linalg.norm(dp, axis=1), 1e-10)
    dp = np.divide(dp, pdist[:, np.newaxis])
    F = k * (length - pdist)
    F[-1] *= k_mouse #Have separate force to mouse, or just set pos

    for node in range(1, n_nodes):
        pdotdot[node] = (F[node] * dp[node] - F[node - 1] * dp[node - 1] + m * g) / m
        pdot[node] = 0.99 * pdot[node] + dt * pdotdot[node]
        p[node] += dt * pdot[node]
        #print(F[node - 1] * dp[node - 1], m * g)
    p[-1] = mp

    screen.fill((0, 0, 0))
    for node in range(n_nodes):
        pygame.draw.circle(screen, (255, 0, 0), (int(p[node, 0] + 0.5),
                                                 int(p[node, 1] + 0.5)), 3)
       #  scale = 100
       #  pygame.draw.line(screen, (0, 255, 0),
       #                   (int(p[node, 0] + 0.5), int(p[node, 1] + 0.5)),
       #                   (int(p[node, 0] - F[node - 1] * dp[node - 1, 0] / scale), int(p[node, 1] - F[node - 1] * dp[node - 1, 1] / scale)))
       #  scale = 100
       #  pygame.draw.line(screen, (0, 0, 255),
       #                   (int(p[node, 0] + 0.5), int(p[node, 1] + 0.5)),
       #                   (int(p[node, 0] + F[node] * dp[node, 0] / scale), int(p[node, 1] + F[node] * dp[node, 1] / scale)))




    pygame.display.flip()
    clock.tick(fps)
