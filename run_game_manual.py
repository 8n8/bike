""" It runs the navigation game. """

import game_gui as g
import game_functions as f
import evaluate_net


g.main(evaluate_net.init, 0.03, f.manual_update_world, f.world2view)
