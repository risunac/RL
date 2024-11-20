#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:42:25 2024

@author: vmuser
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class SurvivalTracker:
    def __init__(self, initial_population: int):
        """Initialize survival rate tracker."""
        self.initial_population = initial_population
        self.time_periods = []
        self.survival_rates = []
        
    def update(self, living_ants: List, dead_ants: List, current_time: int) -> Dict:
        """Update survival statistics for current time period."""
        # Calculate current survival rate
        current_population = len(living_ants)
        survival_rate = (current_population / self.initial_population) * 100
        
        # Update tracking lists
        self.time_periods.append(current_time)
        self.survival_rates.append(survival_rate)
        
        # Return current metrics
        return {
            'Current Survival Rate': f"{survival_rate:.1f}%",
            'Living Ants': current_population
        }

def plot_survival_metrics(survival_tracker, current_episode=None):
    """Creates simple survival rate visualization."""
    plt.figure(figsize=(10, 6))
    
    # Plot survival rate over time
    plt.plot(survival_tracker.time_periods, survival_tracker.survival_rates, 
             color='blue', label='Survival Rate')
    plt.title('Survival Rate Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Survival Rate (%)')
    plt.grid(True)
    
    if current_episode is not None:
        plt.suptitle(f'Episode {current_episode}')
    plt.tight_layout()
    plt.show()

class EpisodeTracker:
    """Tracks survival metrics across multiple episodes"""
    def __init__(self):
        self.episode_metrics = []
        
    def record_episode(self, survival_tracker, episode_num):
        """Record metrics for an episode"""
        if not survival_tracker.survival_rates:
            return
            
        self.episode_metrics.append({
            'episode': episode_num,
            'final_survival_rate': survival_tracker.survival_rates[-1]
        })
    
    def plot_episode_comparison(self):
        """Plot comparison of episodes"""
        if not self.episode_metrics:
            return
            
        episodes = [m['episode'] for m in self.episode_metrics]
        survival_rates = [m['final_survival_rate'] for m in self.episode_metrics]
        
        plt.figure(figsize=(10, 6))
        plt.bar(episodes, survival_rates, color='blue')
        plt.title('Final Survival Rates Across Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Final Survival Rate (%)')
        plt.grid(True)
        plt.show()