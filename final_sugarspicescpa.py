#my most recent basic with trade, after comparative advantage
import pygame
from copy import deepcopy
import copy
import torch
import random
import DQNAgent
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from survival_plotting import SurvivalTracker 
# Constants to define game settings
NUMBER_OF_ACTIONS = 13
WIDTH, HEIGHT = 800, 600
SQUARE_SIZE = 5
REGENERATION_INTERVAL = 1000
trading = True

# Ants
ANT_SIZE = 10
ANT_SPEED = 20
NUM_ANTS = 10
ANT_HEALTH = 100
HEALTH_INCREASE = 10
HEALTH_DECREASE_RATE = 0.1
CRITICAL_HEALTH = 20 
# Sugar
SUGAR_RADIUS = 150
SUGAR_MAX = 100
SUGAR_REGENERATION_RATE = 0.1

#Spice
SPICE_RADIUS = 150
SPICE_MAX = 100
SPICE_REGENERATION_RATE = 0.1

GRID_SIZE = 15
HALF_GRID_SIZE = GRID_SIZE #// 2

# Trade
SPICE_TRADING_DISTANCE = 50
SUGAR_TRADING_DISTANCE = 50
TRADING_ANGLE_THRESHOLD = math.pi / 2

# Color definitions
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Initialize font
pygame.font.init()
FONT = pygame.font.Font(None, 36)

def get_ant_color(health, ant_type):
    """
    Determine the color of an ant based on its health.
    As health decreases, the ant fades to white.
    """
    intensity = int((health / ANT_HEALTH) * 255)
    if ant_type == "sugar":    
        return (255, 255, 255 - intensity)
    else:
        return (255, 0, 255 - intensity)

class Ant:
    def __init__(self, x, y, adv_type, agent):
        self.agent = agent
        self.x = x
        self.y = y
        self.type = adv_type
        self.health_sugar = 100
        self.health_spice = 100
        self.health = min(self.health_sugar, self.health_spice)
        self.direction = random.sample([0, 1/2 * math.pi , math.pi, 3/2 * math.pi], 1)[0]
        self.turn_angle = random.sample([0, 1/2 * math.pi , math.pi, 3/2 * math.pi], 1)[0]
        self.avoid_distance = 20
        self.sugar = 0
        self.spice = 0
        self.trade_stats = {
            'comparative': {'success': 0, 'failed': 0},
            'inverse': {'success': 0, 'failed': 0},
            'swindle': {'success': 0, 'failed': 0}
        }
        self.trade_timeline = []  # Will store (timestamp, trade_type, success) tuples
        if self.type == "sugar":
            self.sugar_skill = 1.5 # Comparative advantage in mining sugar
            self.spice_skill = 1  # Comparative advantage in mining spice
        elif self.type == "spice":
            self.sugar_skill = 1  # Comparative advantage in mining sugar
            self.spice_skill = 1.5  # Comparative advantage in mining spice
        self.sugar_preference = 1
        self.spice_preference = 1
        self.action_history = {
            'exploring': 0,
            'mining_sugar': 0,
            'mining_spice': 0,
            'trading': 0,
            'consuming': 0
        }
        # Performance tracking
        self.total_reward = 0
        self.actions_taken = {i: 0 for i in range(NUMBER_OF_ACTIONS)}  # Track action frequencies
        
        try:
            # Initial state tensor
            self.state = torch.tensor([
                self.x / WIDTH,
                self.y / HEIGHT,
                self.health / ANT_HEALTH,
                self.sugar,
                self.spice,
                0,  # Nearby sugar indicator
                0,  # Nearby spice indicator
                0,  # Can trade indicator
                0,  # Comparative advantage indicator
                self.direction / (2 * math.pi)
            ], dtype=torch.float32).unsqueeze(0).to(device)

        except Exception as e:
            print(f"Error initializing ant: {e}")
            self.health = 0
        
        self.action_state = "exploring"
        
    def get_state_representation(self):
        """Updates and returns the current state representation"""
        if self.sugarscape is None:
            return self.action_state
    
        try:
            nearby_sugar = 0
            nearby_spice = 0
            for sugar in self.sugarscape.sugar_patches:
                if sugar[2] and self.is_within_radius(sugar, SUGAR_RADIUS):
                    nearby_sugar = 1
                    break
            for spice in self.sugarscape.spice_patches:
                if spice[2] and self.is_within_radius(spice, SPICE_RADIUS):
                    nearby_spice = 1
                    break
    
            can_trade = 0
            has_comparative_advantage = 0
            for other_ant in self.sugarscape.ants:
                if other_ant != self and self.can_trade_with(other_ant):
                    can_trade = 1
                    if (abs(self.spice - self.sugar) > 0.2 and 
                        abs(other_ant.sugar - other_ant.spice) > 0.2):
                        has_comparative_advantage = 1
                    break
    
            return torch.tensor([self.x / WIDTH,
                                 self.y / HEIGHT,
                                 self.health / ANT_HEALTH,
                                 self.sugar, self.spice,
                                 nearby_sugar,
                                 nearby_spice,
                                 can_trade,
                                 self.sugar_skill/self.spice_skill,
                                 self.direction / (2 * math.pi)],
                                dtype=torch.float32).unsqueeze(0).to(device)
        
        except Exception as e:
            print(f"Error in get_state_representation: {e}")
            return self.state

    # def act(self, sugarscape):
    #     if self.health <= 0:
    #         return

    #     self.decrease_health()

    #     actions = {
    #         "exploring": self.explore,
    #         "move_up": self.move_up,
    #         "move_down": self.move_down,
    #         "move_left": self.move_left,
    #         "move_right": self.move_right,
    #         "mining_sugar": self.mine_sugar,
    #         "mining_spice": self.mine_spice,
    #         "trading": self.trade,
    #         "trade_sugar_0": self.trade,
    #         "trade_sugar_adv": self.trade,
    #         "trade_spice_0": self.trade,
    #         "trade_spice_adv": self.trade,
    #         "consuming": self.consume_resources
    #     }
    #     # print(self.health)
    #     # print(self.sugar)
    #     # print(self.spice)
    #     # print(self.action_state)
    #     actions[self.action_state](sugarscape)
    #     if self.action_state == "mining_sugar" or self.action_state == "mining_spice":
    #         actions[self.action_state](sugarscape)
        
    #     self.action_state = "trading"
    #     # print(self.action_state)
    #     actions[self.action_state](sugarscape)
    #     # print(self.action_state)
    #     self.action_state = "consuming"
    #     # print(self.action_state)
    #     actions[self.action_state](sugarscape)
    
    def act(self, sugarscape):
        """Main action selection and execution method"""
        try:
            # Update health
            
            # Store sugarscape reference and get state
            self.sugarscape = sugarscape
            self.state = self.get_state_representation()
            
            # Select and execute action
            action = self.agent.select_action(self.state)
            self.actions_taken[action.item()] += 1
            
            reward = self._execute_action(action, sugarscape)
            self.total_reward += reward
            
            # Get next state and store transition
            next_state = self.get_state_representation()
            self.agent.memory.push(
                self.state,
                action.to(device),
                next_state,
                torch.tensor([reward], device=device)
            )
            
            # Update state and optimize
            self.state = next_state
            self.agent.optimize_model()
            self.agent.update_target_network()
            
        except Exception as e:
            print(f"Error in ant action: {e}")
            self.health = 0

    def _execute_action(self, action, sugarscape):
        """Helper method to execute actions and calculate rewards"""
        reward = 0
        
        try:
            if action in [0, 1, 2, 3, 4, 5, 6, 7]:
                # All movement actions
                self.action_history['exploring'] += 1
            if action == 0:  # Explore
                # self.move_up(sugarscape.ants)
                # reward = 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_up(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 1:  # Explore
                # self.move_down(sugarscape.ants)
                # reward = 0.
               
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_down(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 2:  # Explore
                # self.move_left(sugarscape.ants)
                # reward = 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_left(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 3:  # Explore
                # self.move_right(sugarscape.ants)
                # reward = 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_right(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 4:  # Explore
                # self.move_upright(sugarscape.ants)
                # reward = 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_upright(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 5:  # Explore
                # self.move_downright(sugarscape.ants)
                # reward = 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_downright(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 6:  # Explore
                # self.move_upleft(sugarscape.ants)
                # reward = 0.
              
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_upleft(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 7:  # Explore
                # self.move_downleft(sugarscape.ants)
                # reward = 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.move_downleft(sugarscape.ants)
                reward = self.health - temp_health
            elif action == 8 and self.state[0][5] > 0:  # Mine sugar if nearby
                # temp_sugar = deepcopy(self.sugar)
                # self.mine_sugar(sugarscape)
                # reward = (self.sugar - temp_sugar) * 0.
                self.action_history['mining_sugar'] += 1
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.mine_sugar(sugarscape)
                reward = self.health - temp_health
            elif action == 9 and self.state[0][6] > 0:  # Mine spice if nearby
                # temp_spice = deepcopy(self.spice)
                # self.mine_spice(sugarscape)
                # reward = (self.spice - temp_spice) * 0.
                self.action_history['mining_spice'] += 1
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.mine_spice(sugarscape)
                reward = self.health - temp_health
            elif action == 10 and self.state[0][7] > 0:  # Trade if possible
                # temp_sugar_spice = deepcopy(self.spice) + deepcopy(self.sugar)
                # success = self.trade(sugarscape, "sugar")
                # reward = ((self.spice + self.sugar - temp_sugar_spice) if success else -0.2) * 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                success = self.trade(sugarscape, "sugar")
                reward = self.health - temp_health
            elif action in [10, 11] and self.state[0][7] > 0:  # Trading actions
                self.action_history['trading'] += 1
            elif action == 11 and self.state[0][7] > 0:  # Trade if possible
                # temp_sugar_spice = deepcopy(self.spice) + deepcopy(self.sugar)
                # success = self.trade(sugarscape, "spice")
                # reward = ((self.spice + self.sugar - temp_sugar_spice) if success else -0.2) * 0.
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                success = self.trade(sugarscape, "spice")
                reward = self.health - temp_health
            # elif action == 6 and self.state[0][7] > 0:  # Trade if possible
            #     temp_sugar_spice = deepcopy(self.spice) + deepcopy(self.sugar)
            #     success = self.trade(sugarscape)
            #     reward = (self.spice + self.sugar - temp_sugar_spice) if success else -0.3
            # elif action == 6 and self.state[0][7] > 0:  # Trade if possible
            #     temp_sugar_spice = deepcopy(self.spice) + deepcopy(self.sugar)
            #     success = self.trade(sugarscape)
            #     reward = (self.spice + self.sugar - temp_sugar_spice) if success else -0.3
            # elif action == 6 and self.state[0][7] > 0:  # Trade if possible
            #     temp_sugar_spice = deepcopy(self.spice) + deepcopy(self.sugar)
            #     success = self.trade(sugarscape)
            #     reward = (self.spice + self.sugar - temp_sugar_spice) if success else -0.3
            elif action == 12 and (self.sugar > 0 or self.spice > 0):  # Consume
                self.action_history['consuming'] += 1
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.consume_resources(None)
                reward = self.health - temp_health
            else:
                temp_health = deepcopy(self.health)
                self.decrease_health()
                reward = self.health - temp_health -0.2  # Invalid action penalty

            # Additional reward components
            # reward += self.health / 200.0
            # reward -= 0.5 if self.health <= 0 else 0
            
            return reward
        except Exception as e:
            print(f"Error executing action: {e}")
            return -1.0
        
    #moving functions
    def explore(self, sugarscape):
        self.direction = random.sample([0, 1/2 * math.pi , math.pi, 3/2 * math.pi], 1)[0]
        self.move(sugarscape.ants)
        self.detect_resources(sugarscape)
        
    def move_up(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = 6/4 * math.pi
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def move_down(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = 4/8 * math.pi
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))
        
    def move_left(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = math.pi
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))
        
    def move_right(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = 0
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))
        
    def move_upright(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = 7/4 * math.pi
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def move_upleft(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = 5/4 * math.pi
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))
        
    def move_downright(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = 1/4 * math.pi
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))
        
    def move_downleft(self, ants):
        self.detect_nearby_ants(ants)
        self.direction = 3/4 * math.pi
        self.x += ANT_SPEED * math.cos(self.direction)
        self.y += ANT_SPEED * math.sin(self.direction)
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    #mining functions
    def mine_resource(self, resource_type):
        if resource_type == 'sugar':
            mined_amount = self.sugar_skill
            self.sugar += mined_amount
        elif resource_type == 'spice':
            mined_amount = self.spice_skill
            self.spice += mined_amount
        return mined_amount

    def mine_sugar(self, sugarscape):
        for sugar in sugarscape.sugar_patches:
            if sugar[2] and self.is_within_radius(sugar, SUGAR_RADIUS):
                sugar[2] = False  # Make it invisible
                sugar[3] -= 1
                sugarscape.consumed_sugar_patches.append(sugar)  # Add to consumed list
                mined_amount = self.mine_resource('sugar')
                sugarscape.sugar_mined += mined_amount
                return
    
    def mine_spice(self, sugarscape):
        for spice in sugarscape.spice_patches:
            if spice[2] and self.is_within_radius(spice, SPICE_RADIUS):
                spice[2] = False  # Make it invisible
                spice[3] -= 1
                sugarscape.consumed_spice_patches.append(spice)  # Add to consumed list
                mined_amount = self.mine_resource('spice')
                sugarscape.spice_mined += mined_amount
                return

    #trading functions
    def trade(self, sugarscape, resource_type):
        for other_ant in sugarscape.ants:
            if other_ant != self and self.can_trade_with(other_ant):
                if self.trade_resources(other_ant, resource_type):
                    sugarscape.trade_count += 1
                    return
    def record_trade(self, trade_type, success):
        """Record a trade attempt and its outcome"""
        if success:
            self.trade_stats[trade_type]['success'] += 1
        else:
            self.trade_stats[trade_type]['failed'] += 1
        
        # Record in timeline with timestamp
        import time
        self.trade_timeline.append((time.time(), trade_type, success)) 
                    
                    
    def trade_resources(self, other_ant, resource_type, balance_threshold=5):
       
        traded = False
        
        # Basic validation - check if other ant has any resources
        if other_ant.sugar + other_ant.spice == 0:
            return traded
        
        # 4. Survival Swindling (checking this first as it's a survival necessity)
        # If health is critically low, attempt to swindle resources needed for survival
        if self.health <= CRITICAL_HEALTH:
            if self.type == "sugar":
                # Sugar ant critically needs spice
                if other_ant.spice > 0:  # Only try if other ant has spice
                    trade_amount = min(other_ant.spice, self.health_spice/2)  # Take what's needed to survive
                    other_ant.spice -= trade_amount
                    self.spice += trade_amount
                    self.record_trade('swindle', True)
                    return True
            else:  # spice ant
                # Spice ant critically needs sugar
                if other_ant.sugar > 0:  # Only try if other ant has sugar
                    trade_amount = min(other_ant.sugar, self.health_sugar/2)  # Take what's needed to survive
                    other_ant.sugar -= trade_amount
                    self.sugar += trade_amount
                    self.record_trade('swindle', True)
                    return True
                   
            self.record_trade('swindle', False)
            return False
        
        # If not in survival mode, proceed with normal trading strategies
        
        # 1. Comparative Advantage Trading
        # 1. Comparative Advantage Trading
        if self.type == "sugar" and resource_type == "spice":
            if self.sugar >= self.sugar_skill and other_ant.spice >= other_ant.spice_skill:
                trade_amount = min(self.sugar_skill, other_ant.spice_skill)
                if self.sugar >= trade_amount and other_ant.spice >= trade_amount:
                    self.sugar -= trade_amount
                    other_ant.sugar += trade_amount
                    self.spice += trade_amount
                    other_ant.spice -= trade_amount
                    self.record_trade('comparative', True)
                    return True
            self.record_trade('comparative', False)
            return False
            
        elif self.type == "spice" and resource_type == "sugar":
            if self.spice >= self.spice_skill and other_ant.sugar >= other_ant.sugar_skill:
                trade_amount = min(self.spice_skill, other_ant.sugar_skill)
                if self.spice >= trade_amount and other_ant.sugar >= trade_amount:
                    self.spice -= trade_amount
                    other_ant.spice += trade_amount
                    self.sugar += trade_amount
                    other_ant.sugar -= trade_amount
                    self.record_trade('comparative', True)
                    return True
            self.record_trade('comparative', False)
            return False
        
        # 2. Inverse Comparative Advantage
        elif self.type == "sugar" and resource_type == "sugar":
            if self.spice >= self.spice_skill and other_ant.sugar >= other_ant.sugar_skill:
                trade_amount = min(self.spice_skill, other_ant.sugar_skill)
                if self.spice >= trade_amount and other_ant.sugar >= trade_amount:
                    self.spice -= trade_amount
                    other_ant.spice += trade_amount
                    self.sugar += trade_amount
                    other_ant.sugar -= trade_amount
                    self.record_trade('inverse', True)
                    return True
            self.record_trade('inverse', False)
            return False
                    
        elif self.type == "spice" and resource_type == "spice":
            if self.sugar >= self.sugar_skill and other_ant.spice >= other_ant.spice_skill:
                trade_amount = min(self.sugar_skill, other_ant.spice_skill)
                if self.sugar >= trade_amount and other_ant.spice >= trade_amount:
                    self.sugar -= trade_amount
                    other_ant.sugar += trade_amount
                    self.spice += trade_amount
                    other_ant.spice -= trade_amount
                    self.record_trade('inverse', True)
                    return True
            self.record_trade('inverse', False)
            return False
        
        return False  # If no trade conditions were met
        

    
    #consuming functions
    def consume_resources(self, _):
        sugar_consumed = min(self.sugar, self.sugar_preference, ANT_HEALTH - self.health_sugar)
        spice_consumed = min(self.spice, self.spice_preference, ANT_HEALTH - self.health_spice)

        self.sugar -= sugar_consumed
        self.spice -= spice_consumed
    
        self.health_sugar += sugar_consumed
        self.health_spice += spice_consumed
        self.health = max(min(self.health_sugar, self.health_spice), 0)
        
        self.action_state = "exploring"

    #state maintaining functions
    def detect_resources(self, sugarscape):
        for sugar in sugarscape.sugar_patches:
            if sugar[2] and self.is_within_radius(sugar, SUGAR_RADIUS):
                self.action_state = "mining_sugar"
                return

        for spice in sugarscape.spice_patches:
            if spice[2] and self.is_within_radius(spice, SPICE_RADIUS):
                self.action_state = "mining_spice"
                return

    def is_within_radius(self, resource, radius):
        dx = resource[0] - self.x
        dy = resource[1] - self.y
        return math.sqrt(dx ** 2 + dy ** 2) < radius

    def can_trade_with(self, other_ant):
        dx = other_ant.x - self.x
        dy = other_ant.y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return (distance < SPICE_TRADING_DISTANCE or distance < SUGAR_TRADING_DISTANCE) and trading

    def detect_nearby_ants(self, ants):
        for other_ant in ants:
            if other_ant != self:
                dx = other_ant.x - self.x
                dy = other_ant.y - self.y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance < self.avoid_distance:
                    self.direction += math.pi
                    return

    def decrease_health(self):
        self.health_sugar -= HEALTH_DECREASE_RATE
        self.health_sugar = max(0, self.health_sugar)
        
        self.health_spice -= HEALTH_DECREASE_RATE
        self.health_spice = max(0, self.health_spice)
        
        self.health = min(self.health_sugar, self.health_spice)

class SugarScape:
    def __init__(self):
        self.sugar_spots = [(200, 300), (600, 300)]
        self.spice_spots = [(400, 100), (400, 500)]
        self.agent = DQNAgent.DQNAgent(n_observations=10, n_actions=NUMBER_OF_ACTIONS)
        self.ants = [Ant(random.randint(0, WIDTH), random.randint(0, HEIGHT), "sugar", self.agent) for i in range(0,NUM_ANTS)]
        for i in range(0,NUM_ANTS): self.ants.append(Ant(random.randint(0, WIDTH), random.randint(0, HEIGHT), "spice", self.agent))
        self.dead_ants = []
        self.sugar_patches = self.initialize_sugar_patches()
        self.spice_patches = self.initialize_spice_patches()
        self.consumed_sugar_patches = []
        self.consumed_spice_patches = []
        self.last_regeneration_time = pygame.time.get_ticks()
        self.trade_count = 0
        self.trade_counts = []
        self.health_all = []
        self.health_living = []
        self.sugar_mined = 0
        self.spice_mined = 0
        self.trade_counts = []
        self.comparative_trades = []
        self.inverse_trades = []
        self.swindle_trades = []
        self.action_counts = {
            'exploring': [],
            'mining_sugar': [],
            'mining_spice': [],
            'trading': [],
            'consuming': []
        }
        self.metrics = {
            "Avg Health (All)": 0,
            "Avg Health (Living)": 0,
            "Sugar Available": len(self.sugar_patches),
            "Spice Available": len(self.spice_patches),
            "Living Ants": NUM_ANTS,
            "Total Trades": 0,
            "Sugar Mined": 0,
            "Spice Mined": 0
        }
        self.survival_tracker = SurvivalTracker(NUM_ANTS * 2)

    def reset(self):
        self.sugar_spots = [(200, 300), (600, 300)]
        self.spice_spots = [(400, 100), (400, 500)]
        self.ants = [Ant(random.randint(0, WIDTH), random.randint(0, HEIGHT), "sugar", self.agent) for i in range(0,NUM_ANTS)]
        for i in range(0,NUM_ANTS): self.ants.append(Ant(random.randint(0, WIDTH), random.randint(0, HEIGHT), "spice", self.agent))
        self.dead_ants = []
        self.sugar_patches = self.initialize_sugar_patches()
        self.spice_patches = self.initialize_spice_patches()
        self.consumed_sugar_patches = []
        self.consumed_spice_patches = []
        self.last_regeneration_time = pygame.time.get_ticks()
        self.trade_count = 0
        self.trade_counts = []
        self.health_all = []
        self.health_living = []
        self.sugar_mined = 0
        self.spice_mined = 0
        self.trade_counts = []
        self.comparative_trades = []
        self.inverse_trades = []
        self.swindle_trades = []
        self.action_counts = {
            'exploring': [],
            'mining_sugar': [],
            'mining_spice': [],
            'trading': [],
            'consuming': []
        }
        self.metrics = {
            "Avg Health (All)": 0,
            "Avg Health (Living)": 0,
            "Sugar Available": len(self.sugar_patches),
            "Spice Available": len(self.spice_patches),
            "Living Ants": NUM_ANTS,
            "Total Trades": 0,
            "Sugar Mined": 0,
            "Spice Mined": 0
        }
        self.survival_tracker = SurvivalTracker(NUM_ANTS * 2)

    def initialize_sugar_patches(self):
        patches = []
        for (x, y) in self.sugar_spots:
            for n in range(SUGAR_MAX):
                square_x = x + (n % GRID_SIZE) * SQUARE_SIZE - HALF_GRID_SIZE * SQUARE_SIZE
                square_y = y + (n // GRID_SIZE) * SQUARE_SIZE - HALF_GRID_SIZE * SQUARE_SIZE
                patches.append([square_x, square_y, True, SQUARE_SIZE])
        return patches
    
    def initialize_spice_patches(self):
        patches = []
        for (x, y) in self.spice_spots:
            for n in range(SPICE_MAX):
                square_x = x + (n % GRID_SIZE) * SQUARE_SIZE - HALF_GRID_SIZE * SQUARE_SIZE
                square_y = y + (n // GRID_SIZE) * SQUARE_SIZE - HALF_GRID_SIZE * SQUARE_SIZE
                patches.append([square_x, square_y, True, SQUARE_SIZE])
        return patches

    def update(self):
        self.trade_count = 0
        for ant in self.ants[:]:
            if ant.health > 0:
                ant.act(self)
            else:
                self.dead_ants.append(ant)
                self.ants.remove(ant)

        # Calculate and update health metrics
        all_healths = [ant.health for ant in self.ants] + [0 for _ in self.dead_ants]
        living_healths = [ant.health for ant in self.ants if ant.health > 0]
        
        avg_health_all = sum(all_healths) / (len(self.ants) + len(self.dead_ants)) if all_healths else 0
        avg_health_living = sum(living_healths) / len(living_healths) if living_healths else 0
        comparative_total = sum(ant.trade_stats['comparative']['success'] for ant in self.ants)
        inverse_total = sum(ant.trade_stats['inverse']['success'] for ant in self.ants)
        swindle_total = sum(ant.trade_stats['swindle']['success'] for ant in self.ants)
        current_actions = {
            'exploring': sum(ant.action_history['exploring'] for ant in self.ants),
            'mining_sugar': sum(ant.action_history['mining_sugar'] for ant in self.ants),
            'mining_spice': sum(ant.action_history['mining_spice'] for ant in self.ants),
            'trading': sum(ant.action_history['trading'] for ant in self.ants),
            'consuming': sum(ant.action_history['consuming'] for ant in self.ants)
        }
        survival_metrics = self.survival_tracker.update(self.ants, self.dead_ants, len(self.trade_counts))
       
        
        for action_type in self.action_counts:
            self.action_counts[action_type].append(current_actions[action_type])
        self.comparative_trades.append(comparative_total)
        self.inverse_trades.append(inverse_total)
        self.swindle_trades.append(swindle_total)
        self.metrics["Avg Health (All)"] = avg_health_all
        self.metrics["Avg Health (Living)"] = avg_health_living
        
        # Store health metrics for plotting
        self.health_all.append(avg_health_all)
        self.health_living.append(avg_health_living)
        
        self.metrics["Sugar Available"] = len([s for s in self.sugar_patches if s[2]])
        self.metrics["Spice Available"] = len([s for s in self.spice_patches if s[2]])
        self.metrics["Living Ants"] = len(self.ants)
        self.metrics["Total Trades"] += self.trade_count
        self.metrics["Sugar Mined"] = self.sugar_mined
        self.metrics["Spice Mined"] = self.spice_mined
        self.trade_counts.append(self.trade_count + (self.trade_counts[-1] if self.trade_counts else 0))
        self.metrics.update(survival_metrics) 
        self.regenerate_resources()

    def regenerate_resources(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_regeneration_time > REGENERATION_INTERVAL:
            # Regenerate sugar
            for sugar in self.sugar_patches:
                if not sugar[2] and sugar in self.consumed_sugar_patches:  # Only regenerate if actually consumed
                    if random.random() < SUGAR_REGENERATION_RATE:
                        sugar[2] = True  # Respawn the sugar
                        self.consumed_sugar_patches.remove(sugar)
            
            # Regenerate spice
            for spice in self.spice_patches:
                if not spice[2] and spice in self.consumed_spice_patches:  # Only regenerate if actually consumed
                    if random.random() < SPICE_REGENERATION_RATE:
                        spice[2] = True  # Respawn the spice
                        self.consumed_spice_patches.remove(spice)
                
            self.last_regeneration_time = current_time

def plot_health_and_trades(trade_counts, health_all, health_living, comparative_trades, inverse_trades, swindle_trades, action_counts):
    fig = plt.figure(figsize=(15, 15)) 
    
    # Plot 1 and 2 remain the same
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(trade_counts, color='blue', label='Total Trades')
    ax1.set_title('Cumulative Trades Over Time')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Total Trades')
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(health_all, color='green', label='Avg Health (All)')
    ax2.plot(health_living, color='red', label='Avg Health (Living)')
    ax2.set_title('Average Health Over Time')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Average Health')
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
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
    
    # Create period labels
    period_labels = [f'Steps\n{i*steps_per_bin}-{(i+1)*steps_per_bin}' 
                    for i in range(num_bins)]
    
    ax3.bar([x - bar_width for x in indices], comp_binned,
            bar_width, label='Comparative', color='green')
    ax3.bar([x for x in indices], inv_binned,
            bar_width, label='Inverse', color='red')
    ax3.bar([x + bar_width for x in indices], swin_binned,
            bar_width, label='Swindle', color='orange')
    
    ax3.set_title('Average Trade Types Over Time')
    ax3.set_xlabel('Time Periods')
    ax3.set_ylabel('Average Number of Trades')
    ax3.set_xticks(range(num_bins))
    ax3.set_xticklabels(period_labels)
    ax3.legend()
    # Plot 4 remains the same
    ax4 = plt.subplot(2, 2, 4)
    total_trades = sum([sum(comparative_trades), sum(inverse_trades), sum(swindle_trades)])
    if total_trades > 0:
        trade_props = [sum(comparative_trades)/total_trades, 
                      sum(inverse_trades)/total_trades, 
                      sum(swindle_trades)/total_trades]
        ax4.pie(trade_props, 
                labels=['Comparative', 'Inverse', 'Swindle'],
                colors=['green', 'red', 'orange'],
                autopct='%1.1f%%')
        ax4.set_title('Trade Type Distribution')
    ax5 = plt.subplot(3, 2, 5)
    num_bins = 5  # Fixed number of bins
    
    def bin_data(data, num_bins):
        if not data:
            return np.zeros(num_bins)
        
        # Convert to numpy array and reshape
        data_array = np.array(data)
        # Create exactly num_bins by averaging
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
    
    # Plot bars for each action type
    for i, (action_type, color) in enumerate(colors.items()):
        if action_type in action_counts:
            binned_data = bin_data(action_counts[action_type], num_bins)
            offset = (i - len(colors)/2) * bar_width
            ax5.bar(indices + offset, binned_data, bar_width, 
                   label=action_type.capitalize(), color=color)
    
    period_labels = [f'Steps\n{i}-{i+1}' for i in range(num_bins)]
    ax5.set_title('Average Actions Over Time')
    ax5.set_xlabel('Time Periods')
    ax5.set_ylabel('Average Number of Actions')
    ax5.set_xticks(indices)
    ax5.set_xticklabels(period_labels)
    ax5.legend()


    plt.tight_layout()
    plt.show()

def draw_metrics(surface, metrics):
    y_offset = 10
    for metric, value in metrics.items():
        text = FONT.render(f"{metric}: {int(value)}" if isinstance(value, float) else f"{metric}: {value}", True, BLACK)
        surface.blit(text, (10, y_offset))
        y_offset += 30

def draw_sugarscape(surface, sugarscape):
    surface.fill(WHITE)
    for ant in sugarscape.ants:
        pygame.draw.circle(surface, get_ant_color(ant.health, ant.type), (int(ant.x), int(ant.y)), ANT_SIZE)
    for sugar in sugarscape.sugar_patches:
        if sugar[2]:
            pygame.draw.rect(surface, GREEN, pygame.Rect(sugar[0], sugar[1], sugar[3], sugar[3]))
    for spice in sugarscape.spice_patches:
        if spice[2]:
            pygame.draw.rect(surface, RED, pygame.Rect(spice[0], spice[1], spice[3], spice[3]))

    # Draw trade lines (making them slimmer)
    for i, ant in enumerate(sugarscape.ants):
        for other_ant in sugarscape.ants[i+1:]:
            if (ant.health > 0 and other_ant.health > 0 and
                math.sqrt((ant.x - other_ant.x) ** 2 + (ant.y - other_ant.y) ** 2) < SPICE_TRADING_DISTANCE):
                pygame.draw.line(surface, BLUE, (ant.x, ant.y), (other_ant.x, other_ant.y), 1)  # Change line width to 1

    draw_metrics(surface, sugarscape.metrics)
    pygame.display.flip()