import random

# -------------------------------------------------------
# TOURNAMENT SELECTION
# -------------------------------------------------------

def select_parent(population, fitness_key, tournament_size=3):
    """
    Tournament Selection
    """

    contestants = random.sample(population, tournament_size)
    contestants.sort(key=lambda x: x[fitness_key], reverse=True)

    return contestants[0]