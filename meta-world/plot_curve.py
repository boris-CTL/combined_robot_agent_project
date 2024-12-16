

# Load the csv from wandb
def load_csv(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df


# Plot the curve
def plot_curve(df, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="darkgrid")

    # Right shift the pretrain curve 20
    # df['ppo_reach-v2 - rollout/ep_rew_mean'] = df['ppo_reach-v2 - rollout/ep_rew_mean'].shift(-20)

    # Right shift the pretrain curve 20
    df['ppo_reach-v2_0321_steps2e5_with_pretrain - rollout/ep_rew_mean'] = df['ppo_reach-v2_0321_steps2e5_with_pretrain - rollout/ep_rew_mean'].shift(20)

    sns.lineplot(x='global_step', y='ppo_reach-v2 - rollout/ep_rew_mean', data=df, label='ppo_reach-v2')

    sns.lineplot(x='global_step', y='ppo_reach-v2_0321_steps2e5_with_pretrain - rollout/ep_rew_mean', data=df, label='ppo_reach-v2_0321_steps2e5_with_pretrain')

    # Let x-axis show by 25k, 50k, 75k, 100k, 125k, 150k, 175k, 200k
    plt.xticks([25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000], ['25k', '50k', '75k', '100k', '125k', '150k', '175k', '200k'])

    # x-axis title
    plt.xlabel('Global Steps')
    # y-axis title
    plt.ylabel('Rollout Ep Rew Mean')

    plt.title(title)
    plt.savefig(save_path)
    # plt.show()


# Main function
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='reach-v2.csv')
    # parser.add_argument('--hue', type=str, default='algo')
    parser.add_argument('--title', type=str, default='Reach-v2')
    parser.add_argument('--save_path', type=str, default='reach-v2_curve.png')
    args = parser.parse_args()
    df = load_csv(args.file_path)
    plot_curve(df, args.title, args.save_path)


if __name__ == '__main__':
    main()