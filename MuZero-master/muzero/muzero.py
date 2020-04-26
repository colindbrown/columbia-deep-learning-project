from config import MuZeroConfig, make_vertex_cover_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.replay_buffer import ReplayBuffer
from training.training import train_network
from matplotlib import pyplot as plt


def muzero(config: MuZeroConfig):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    """
    network =config.new_network()
    storage = SharedStorage(network, config.uniform_network(), config.new_optimizer(network))
    replay_buffer = ReplayBuffer(config)

    train_scores = []
    eval_scores = []

    for loop in range(config.nb_training_loop):
        print("Training loop", loop)
        score_train = run_selfplay(config, storage, replay_buffer, config.nb_episodes)
        train_network(config, storage, replay_buffer, config.nb_epochs)

        print("Train score:", score_train)
        score_eval = run_eval(config, storage, 50)
        print("Eval score:", score_eval)
        train_scores.append(score_train)
        eval_scores.append(score_eval)
        print(f"MuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")

    plt.figure(1)
    plt.plot(train_scores)
    plt.plot(eval_scores)
    plt.title('MuZero Average Rewards')
    plt.xlabel('MuZero Iterations (Train/Eval)')
    plt.ylabel('Reward Score')
    plt.legend(['Train score','Eval score'])
    plt.show()

    return storage.latest_network()


if __name__ == '__main__':
    config = make_vertex_cover_config()
    muzero(config)
