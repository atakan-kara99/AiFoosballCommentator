from src.model.entities.touch import Touch
import numpy

def test_parsing_touch_to_state():
    touch_wall = Touch(dict())
    touch_wall.player = "WALL"
    touch_wall.team_id = None
    touch_wall.time = 0
    touch_wall.position = (0.96, 0.341)
    touch_wall.direction = 110
    touch_wall.speed = 3.2
    touch_wall.goal = False
    touch_wall.throw_in = False

    print(touch_wall.__dict__)
    print("\n")

    state_wall = touch_wall.toGameState()
    print(state_wall.__dict__)
    print("\n")

    print(state_wall.to_normalized_vector())
    print("\n")


    touch_player = Touch(dict())
    touch_player.player = "RCM"
    touch_player.team_id = 0
    touch_player.time = 0
    touch_player.position = (0.45, 0.42)
    touch_player.direction = 33
    touch_player.speed = 10
    touch_player.goal = False
    touch_player.throw_in = False

    print(touch_player.__dict__)
    print("\n")


    state_player = touch_player.toGameState()
    print(state_player.__dict__)
    print("\n")

    print(state_player.to_normalized_vector())
    print("\n")



    touch_goal = Touch(dict())
    touch_goal.goal = True
    touch_goal.team_id = 0

    print(touch_goal.__dict__)
    print("\n")
    state_goal = touch_goal.toGameState()
    print(state_goal.__dict__)
    print("\n")

    print(state_goal.to_normalized_vector())
    print("\n")

    touch_throw_in = Touch(dict())
    touch_throw_in.throw_in = True
    touch_throw_in.team_id = 1
    touch_throw_in.position = (0.5, 0.99)
    touch_throw_in.direction = 90
    touch_throw_in.speed = 2.5

    print(touch_throw_in.__dict__)
    print("\n")
    state_throw_in = touch_throw_in.toGameState()
    print(state_throw_in.__dict__)
    print("\n")

    print(state_throw_in.to_normalized_vector(), "\n")

    sequence = numpy.concatenate([state_wall.to_normalized_vector(), state_player.to_normalized_vector(), state_goal.to_normalized_vector()])
    print(sequence)


if __name__ == "__main__":
    test_parsing_touch_to_state()
    
