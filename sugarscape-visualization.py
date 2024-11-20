import SugarSpiceScape
from SugarSpiceScape import WIDTH, HEIGHT, SugarScape, draw_sugarscape
import DQNAgent
import pygame
import random
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from survival_plotting import SurvivalTracker, plot_survival_metrics, EpisodeTracker

def plot_trades(trade_counts):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(trade_counts, color='blue', label='Total Trades')
    plt.title('Cumulative Trades Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Total Trades')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_health(health_all, health_living):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(health_all, color='green', label='Avg Health (All)')
    plt.plot(health_living, color='red', label='Avg Health (Living)')
    plt.title('Average Health Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Health')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trade_types(comparative_trades, inverse_trades, swindle_trades):
    fig = plt.figure(figsize=(10, 6))
    num_bins = 5  # Reduced for clearer periods
    total_steps = len(comparative_trades)
    steps_per_bin = total_steps // num_bins
    
    def bin_data(data, num_bins):
        bin_size = max(len(data) // num_bins, 1)
        binned_data = []
        for i in range(0, len(data), bin_size):
            bin_avg = sum(data[i:i+bin_size]) / len(data[i:i+bin_size])
            binned_data.append(bin_avg)
        return binned_data

    comp_binned = bin_data(comparative_trades, num_bins)
    inv_binned = bin_data(inverse_trades, num_bins)
    swin_binned = bin_data(swindle_trades, num_bins)
    
    bar_width = 0.25
    indices = range(len(comp_binned))
    
    period_labels = [f'Steps\n{i*steps_per_bin}-{(i+1)*steps_per_bin}' 
                    for i in range(num_bins)]
    
    plt.bar([x - bar_width for x in indices], comp_binned,
            bar_width, label='Comparative', color='green')
    plt.bar([x for x in indices], inv_binned,
            bar_width, label='Inverse', color='red')
    plt.bar([x + bar_width for x in indices], swin_binned,
            bar_width, label='Swindle', color='orange')
    
    plt.title('Average Trade Types Over Time')
    plt.xlabel('Time Periods')
    plt.ylabel('Average Number of Trades')
    plt.xticks(range(num_bins), period_labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trade_distribution(comparative_trades, inverse_trades, swindle_trades):
    fig = plt.figure(figsize=(8, 8))
    total_trades = sum([sum(comparative_trades), sum(inverse_trades), sum(swindle_trades)])
    if total_trades > 0:
        trade_props = [sum(comparative_trades)/total_trades, 
                      sum(inverse_trades)/total_trades, 
                      sum(swindle_trades)/total_trades]
        plt.pie(trade_props, 
                labels=['Comparative', 'Inverse', 'Swindle'],
                colors=['green', 'red', 'orange'],
                autopct='%1.1f%%')
        plt.title('Trade Type Distribution')
    plt.tight_layout()
    plt.show()

def plot_actions(action_counts):
    fig = plt.figure(figsize=(12, 6))
    num_bins = 5  # Fixed number of bins
    
    def bin_data(data, num_bins):
        if not data:
            return np.zeros(num_bins)
        data_array = np.array(data)
        splits = np.array_split(data_array, num_bins)
        return np.array([np.mean(split) if len(split) > 0 else 0 for split in splits])

    colors = {
        'exploring': 'blue',
        'mining_sugar': 'green',
        'mining_spice': 'red',
        'trading': 'purple',
        'consuming': 'orange'
    }
    
    bar_width = 0.15
    indices = np.arange(num_bins)
    
    for i, (action_type, color) in enumerate(colors.items()):
        if action_type in action_counts:
            binned_data = bin_data(action_counts[action_type], num_bins)
            offset = (i - len(colors)/2) * bar_width
            plt.bar(indices + offset, binned_data, bar_width, 
                   label=action_type.capitalize(), color=color)
    
    period_labels = [f'Steps\n{i}-{i+1}' for i in range(num_bins)]
    plt.title('Average Actions Over Time')
    plt.xlabel('Time Periods')
    plt.ylabel('Average Number of Actions')
    plt.xticks(indices, period_labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_metrics(trade_counts, health_all, health_living, comparative_trades, 
                    inverse_trades, swindle_trades, action_counts):
    plot_trades(trade_counts)
    plot_health(health_all, health_living)
    plot_trade_types(comparative_trades, inverse_trades, swindle_trades)
    plot_trade_distribution(comparative_trades, inverse_trades, swindle_trades)
    plot_actions(action_counts)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    sugarscape = SugarScape()
    clock = pygame.time.Clock()
    episode_tracker = EpisodeTracker()
    num_episodes = 20
    steps_per_episode = 25000
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        step = 0
        running = True
        
        while running and len(sugarscape.ants) > 0 and step < steps_per_episode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
    
            sugarscape.update()
            draw_sugarscape(screen, sugarscape)
            clock.tick(1000)
            step += 1
            
        print(f"Episode {episode + 1} completed with {step} steps")
        
        # Plot individual figures after each episode
        plot_all_metrics(
            sugarscape.trade_counts,
            sugarscape.health_all,
            sugarscape.health_living,
            sugarscape.comparative_trades,
            sugarscape.inverse_trades,
            sugarscape.swindle_trades,
            sugarscape.action_counts
        )
        plot_survival_metrics(sugarscape.survival_tracker, current_episode=episode + 1)
        sugarscape.reset()
    
    pygame.quit()

if __name__ == "__main__":
    main()