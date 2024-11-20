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

def get_ant_color(health, specialization):
    """
    Determine the color of an ant based on its health and specialization.
    Sugar specialists fade to pink as health decreases; spice specialists fade to purple.
    """
    intensity = int((health / ANT_HEALTH) * 255)
    
    if specialization == "sugar":
        # Fade from pink to white as health decreases
        return (255, 100, 100 + intensity)
    elif specialization == "spice":
        # Fade from purple to white as health decreases
        return (150, 0, 150 + intensity)
    else:
        # Default color if no specialization is given
        return (200, 200, 200)


class Ant:
    def __init__(self, x, y, agent, is_sugar_specialist):
        self.agent = agent
        self.x = x
        self.y = y
        self.health_sugar = 100
        self.health_spice = 100
        self.health = min(self.health_sugar, self.health_spice)
        self.direction = random.sample([0, 1/2 * math.pi, math.pi, 3/2 * math.pi], 1)[0]
        self.turn_angle = random.sample([0, 1/2 * math.pi, math.pi, 3/2 * math.pi], 1)[0]
        self.avoid_distance = 20
        self.sugar = 0
        self.spice = 0
        self.sugarscape = None  # Initialize sugarscape reference
        
        # Specialist initialization
        self.is_sugar_specialist = is_sugar_specialist
        self.is_spice_specialist = not is_sugar_specialist
        if self.is_sugar_specialist:
            self.sugar_mining_skill = random.uniform(1.0, 1.5)  # Comparative advantage in sugar
            self.spice_mining_skill = random.uniform(0.5, 1.0)
        else:  # Spice specialist
            self.sugar_mining_skill = random.uniform(0.5, 1.0)
            self.spice_mining_skill = random.uniform(1.0, 1.5)  # Comparative advantage in spice
            
        self.sugar_preference = 1
        self.spice_preference = 1
        
        # Performance tracking
        self.total_reward = 0
        self.actions_taken = {i: 0 for i in range(NUMBER_OF_ACTIONS)}
        
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
                self.sugar_mining_skill/self.spice_mining_skill,  # Comparative advantage indicator
                self.direction / (2 * math.pi)
            ], dtype=torch.float32).unsqueeze(0).to(device)

        except Exception as e:
            print(f"Error initializing ant: {e}")
            self.health = 0
            
        self.action_state = "exploring"

    def get_state_representation(self):
        """Updates and returns the current state representation"""
        if self.sugarscape is None:
            return self.state
        
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
                    if (self.is_sugar_specialist and other_ant.is_spice_specialist) or \
                       (self.is_spice_specialist and other_ant.is_sugar_specialist):
                        has_comparative_advantage = 1
                    break
            
            return torch.tensor([
                self.x / WIDTH,
                self.y / HEIGHT,
                self.health / ANT_HEALTH,
                self.sugar,
                self.spice,
                nearby_sugar,
                nearby_spice,
                can_trade,
                self.sugar_mining_skill/self.spice_mining_skill,
                self.direction / (2 * math.pi)
            ], dtype=torch.float32).unsqueeze(0).to(device)
            
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
                
                temp_health = deepcopy(self.health)
                self.decrease_health()
                self.mine_sugar(sugarscape)
                reward = self.health - temp_health
            elif action == 9 and self.state[0][6] > 0:  # Mine spice if nearby
                # temp_spice = deepcopy(self.spice)
                # self.mine_spice(sugarscape)
                # reward = (self.spice - temp_spice) * 0.
                
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
            mined_amount = self.sugar_mining_skill
            self.sugar += mined_amount
        elif resource_type == 'spice':
            mined_amount = self.spice_mining_skill
            self.spice += mined_amount
        return mined_amount

    def mine_sugar(self, sugarscape):
        for sugar in sugarscape.sugar_patches:
            if sugar[2] and self.is_within_radius(sugar, SUGAR_RADIUS):
                sugar[2] = False
                sugar[3] -= 1
                sugarscape.consumed_sugar_patches.append(sugar)
                mined_amount = self.mine_resource('sugar')  # Use sugar skill
                sugarscape.sugar_mined += mined_amount
                return

    def mine_spice(self, sugarscape):
        for spice in sugarscape.spice_patches:
            if spice[2] and self.is_within_radius(spice, SPICE_RADIUS):
                spice[2] = False
                spice[3] -= 1
                sugarscape.consumed_spice_patches.append(spice)
                mined_amount = self.mine_resource('spice')  # Use spice skill
                sugarscape.spice_mined += mined_amount
                return

    #trading functions
    #def trade(self, sugarscape, resource_type):
     #   for other_ant in sugarscape.ants:
      #      if other_ant != self and self.can_trade_with(other_ant):
       #         if self.trade_resources(other_ant, resource_type):
        #            sugarscape.trade_count += 1
         #           return


    def trade_resources(self, other_ant, resource_type):
        try:
            traded = False
            balance_threshold = 0.1
    
            # Compute initial total resources for conservation validation
            initial_total = (self.sugar + self.spice + 
                            other_ant.sugar + other_ant.spice)
    
            # Pure Comparative Advantage trading
            if resource_type == "spice" and self.is_sugar_specialist and self.sugar >= self.sugar_mining_skill and other_ant.spice >= other_ant.spice_mining_skill:
                # Sugar specialist trading sugar for spice
                trade_amount = min(
                    self.sugar_mining_skill,
                    self.sugar,
                    other_ant.spice,
                    10
                )
                if trade_amount > 0:
                    self.sugar -= trade_amount
                    other_ant.sugar += trade_amount
                    self.spice += trade_amount
                    other_ant.spice -= trade_amount
                    traded = True
    
            elif resource_type == "sugar" and self.is_spice_specialist and self.spice >= self.spice_mining_skill and other_ant.sugar >= other_ant.sugar_mining_skill:
                # Spice specialist trading spice for sugar
                trade_amount = min(
                    self.spice_mining_skill,
                    self.spice,
                    other_ant.sugar,
                    10
                )
                if trade_amount > 0:
                    self.spice -= trade_amount
                    other_ant.spice += trade_amount
                    self.sugar += trade_amount
                    other_ant.sugar -= trade_amount
                    traded = True
    
            # For non-specialist trades or when comparative advantage trade fails
            elif not traded:
                if abs(self.spice - self.sugar) < balance_threshold and abs(other_ant.sugar - other_ant.spice) < balance_threshold:
                    # Balanced trade
                    trade_amount = min(1, (self.sugar + self.spice) * 0.1)
                    if trade_amount > 0 and self.spice >= trade_amount and other_ant.sugar >= trade_amount:
                        self.spice -= trade_amount
                        other_ant.spice += trade_amount
                        self.sugar += trade_amount
                        other_ant.sugar -= trade_amount
                        traded = True
                else:
                    # Swindling as last resort
                    if np.random.random() < 0.45:
                        trade_amount = 1
                        if resource_type == "spice" and self.spice >= trade_amount:
                            self.spice -= trade_amount
                            other_ant.spice += trade_amount
                            traded = True
                        elif resource_type == "sugar" and self.sugar >= trade_amount:
                            self.sugar -= trade_amount
                            other_ant.sugar += trade_amount
                            traded = True
    
            if traded:
                # Resource conservation validation
                final_total = (self.sugar + self.spice + 
                              other_ant.sugar + other_ant.spice)
                assert abs(initial_total - final_total) < 1e-6, "Resources not conserved in trade"
    
                # Post-trade operations
                self.consume_resources(None)
                other_ant.consume_resources(None)
                self.calculate_wealth()  # You'll need to add this method if you want to track wealth
                other_ant.calculate_wealth()
    
            return traded
            
        except Exception as e:
            print(f"Error in trade_resources: {e}")
            return False

    def calculate_wealth(self):
        """Calculate the ant's total wealth"""
        self.wealth = self.sugar + self.spice
        return self.wealth

    def can_trade_with(self, other_ant):
        """Check if this ant can trade with another ant"""
        dx = other_ant.x - self.x
        dy = other_ant.y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        # Only allow trading if ants are within range and trading is enabled
        basic_conditions = (distance < SPICE_TRADING_DISTANCE or distance < SUGAR_TRADING_DISTANCE) and trading
        
        # Check for complementary specializations
        complementary_specialists = (
            (self.is_sugar_specialist and other_ant.is_spice_specialist) or
            (self.is_spice_specialist and other_ant.is_sugar_specialist)
        )
        
        return basic_conditions and complementary_specialists

    def trade(self, sugarscape, resource_type):
        """Initiate trading with nearby ants"""
        for other_ant in sugarscape.ants:
            if other_ant != self and self.can_trade_with(other_ant):
                if self.trade_resources(other_ant, resource_type):
                    sugarscape.trade_count += 1
                    return True
        return False
       
        
    
  #  def trade_resources(self, other_ant, resource_type):
   #     traded = False
    #    if self.sugar + self.spice == 0 or other_ant.sugar + other_ant.spice == 0:
     #       return traded
      #  if resource_type == "sugar":
       #     if self.spice >= self.spice_skill and other_ant.sugar >= other_ant.sugar_skill:
                # trade_amount = min((self.spice - self.sugar) * 0.7, (other_ant.sugar - other_ant.spice) * 0.7, 10)
        #        trade_amount = self.spice_skill
         #       self.spice -= trade_amount
          #      other_ant.spice += trade_amount
           #     self.sugar += trade_amount
            #    other_ant.sugar -= trade_amount
             #   traded = True
     #   if resource_type == "spice":
      #      if self.sugar >= self.sugar_skill and other_ant.spice >= other_ant.spice_skill:
       #         trade_amount = self.sugar_skill
        #        self.sugar -= trade_amount
         #       other_ant.sugar += trade_amount
          #      self.spice += trade_amount
           #     other_ant.spice -= trade_amount
            #    traded = True
           # return traded

    

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

    #def can_trade_with(self, other_ant):
     #   dx = other_ant.x - self.x
      #  dy = other_ant.y - self.y
       # distance = math.sqrt(dx ** 2 + dy ** 2)
        #return (distance < SPICE_TRADING_DISTANCE or distance < SUGAR_TRADING_DISTANCE) and trading

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
        self.ants = [
            Ant(random.randint(0, WIDTH), random.randint(0, HEIGHT), self.agent, i % 2 == 0) 
            for i in range(NUM_ANTS)
        ]
        for ant in self.ants:
            ant.sugarscape = self
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

    def reset(self):
        self.sugar_spots = [(200, 300), (600, 300)]
        self.spice_spots = [(400, 100), (400, 500)]
        self.ants = [
            Ant(random.randint(0, WIDTH), random.randint(0, HEIGHT), self.agent, i % 2 == 0)
            for i in range(NUM_ANTS)
        ]

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
        
        self.regenerate_resources()

    def regenerate_resources(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_regeneration_time > REGENERATION_INTERVAL:
            # Regenerate sugar
            for sugar in self.sugar_patches:
                if not sugar[2]:  # If the sugar patch is consumed
                    if random.random() < SUGAR_REGENERATION_RATE:
                        sugar[2] = True  # Respawn the sugar
                        if sugar in self.consumed_sugar_patches:
                            self.consumed_sugar_patches.remove(sugar)
            
            # Regenerate spice
            for spice in self.spice_patches:
                if not spice[2]:  # If the spice patch is consumed
                    if random.random() < SPICE_REGENERATION_RATE:
                        spice[2] = True
                        if spice in self.consumed_spice_patches:
                            self.consumed_spice_patches.remove(spice)
            
            self.last_regeneration_time = current_time

def plot_health_and_trades(trade_counts, health_all, health_living):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot cumulative trades
    ax1.plot(trade_counts, color='blue', label='Total Trades')
    ax1.set_title('Cumulative Trades Over Time')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Total Trades')
    ax1.legend()

    # Plot average healths
    ax2.plot(health_all, color='green', label='Avg Health (All)')
    ax2.plot(health_living, color='red', label='Avg Health (Living)')
    ax2.set_title('Average Health Over Time')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Average Health')
    ax2.legend()

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
        pygame.draw.circle(surface, get_ant_color(ant.health, ant.is_sugar_specialist), (int(ant.x), int(ant.y)), ANT_SIZE)
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