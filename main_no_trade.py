#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:38:29 2024
@author: vmuser
"""
import pygame
from no_trade_learn import WIDTH, HEIGHT, SugarScape, draw_sugarscape
import matplotlib.pyplot as plt
import numpy as np

# Import the survival tracking classes
from surv import SurvivalTracker, EpisodeTracker, plot_survival_metrics

class MetricsTracker:
    def __init__(self):
        self.health_all_history = []
        self.health_living_history = []
        self.action_counts_history = {
            'exploring': [],
            'mining_sugar': [],
            'mining_spice': [],
            'consuming': []
        }

    def update(self, sugarscape):
        # Track health metrics
        all_healths = [ant.health for ant in sugarscape.ants] + [0 for _ in sugarscape.dead_ants]
        living_healths = [ant.health for ant in sugarscape.ants if ant.health > 0]
        
        avg_health_all = sum(all_healths) / (len(sugarscape.ants) + len(sugarscape.dead_ants)) if all_healths else 0
        avg_health_living = sum(living_healths) / len(living_healths) if living_healths else 0
        
        self.health_all_history.append(avg_health_all)
        self.health_living_history.append(avg_health_living)
        
        # Track action counts
        for action_type in self.action_counts_history:
            count = sum(ant.action_history[action_type] for ant in sugarscape.ants)
            self.action_counts_history[action_type].append(count)

def plot_health(health_all, health_living):
    """Plot average health metrics over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(health_all, color='green', label='Avg Health (All)')
    plt.plot(health_living, color='red', label='Avg Health (Living)')
    plt.title('Average Health Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Health')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_actions(action_counts):
    """Plot distribution of actions over time"""
    plt.figure(figsize=(12, 6))
    num_bins = 5
    
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
        'consuming': 'orange'
    }
    
    bar_width = 0.2
    indices = np.arange(num_bins)
    
    for i, (action_type, color) in enumerate(colors.items()):
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

def plot_all_metrics(metrics_tracker, survival_tracker, current_episode=None):
    """Plot all metrics including survival rate"""
    plot_health(metrics_tracker.health_all_history, metrics_tracker.health_living_history)
    plot_actions(metrics_tracker.action_counts_history)
    plot_survival_metrics(survival_tracker, current_episode)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    sugarscape = SugarScape()
    clock = pygame.time.Clock()
    episode_tracker = EpisodeTracker()
    metrics_tracker = MetricsTracker()
    
    num_episodes = 20
    steps_per_episode = 2000
    initial_population = len(sugarscape.ants)  # Total number of ants at start
    
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            step = 0
            running = True
            
            # Initialize survival tracker for this episode
            survival_tracker = SurvivalTracker(initial_population)
            
            while running and len(sugarscape.ants) > 0 and step < steps_per_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        raise SystemExit
                
                sugarscape.update()
                metrics_tracker.update(sugarscape)
                
                # Update survival metrics
                survival_stats = survival_tracker.update(
                    sugarscape.ants,
                    sugarscape.dead_ants,
                    step
                )
                
                draw_sugarscape(screen, sugarscape)
                clock.tick(60)
                step += 1
                
            print(f"Episode {episode + 1} completed with {step} steps")
            print(f"Final Survival Rate: {survival_stats['Current Survival Rate']}")
            print(f"Living Ants: {survival_stats['Living Ants']}")
            
            # Record episode metrics
            episode_tracker.record_episode(survival_tracker, episode + 1)
            
            # Plot metrics after each episode
            plot_all_metrics(metrics_tracker, survival_tracker, episode + 1)
            
            # Reset for next episode while keeping trained agent
            sugarscape = sugarscape.reset()
            
    except (KeyboardInterrupt, SystemExit):
        print("\nSimulation interrupted by user")
    finally:
        pygame.quit()
        # Plot final episode comparison
        episode_tracker.plot_episode_comparison()
        # Final metrics plot
        if 'survival_tracker' in locals():
            plot_all_metrics(metrics_tracker, survival_tracker)

if __name__ == "__main__":
    main()