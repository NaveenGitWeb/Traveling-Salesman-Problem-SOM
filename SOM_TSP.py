import numpy as np
import matplotlib.pyplot as plt

# Initialize the SOM for TSP
class SOM_TSP:
    def __init__(self, num_cities, num_neurons, learning_rate, radius, num_iterations):
        self.num_cities = num_cities
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.radius = radius
        self.num_iterations = num_iterations
        
        # Randomly initialize neuron positions (1D ring in 2D space)
        self.neurons = np.random.rand(self.num_neurons, 2)
    
    def train(self, cities):
        for iteration in range(self.num_iterations):
            for city in cities:
                # Step 1: Find the winning neuron (closest neuron to the city)
                distances = np.linalg.norm(self.neurons - city, axis=1)
                winner_idx = np.argmin(distances)
                
                # Step 2: Update the neurons in the neighborhood of the winner
                self.update_neurons(city, winner_idx, iteration)
            
            # Optionally reduce the learning rate and radius over time
            self.learning_rate *= 0.995
            self.radius *= 0.995
    
    def update_neurons(self, city, winner_idx, iteration):
        for i in range(self.num_neurons):
            distance_to_winner = abs(i - winner_idx)
            if distance_to_winner <= self.radius:
                # Update neuron position based on the learning rule
                self.neurons[i] += self.learning_rate * (city - self.neurons[i])
    
    def get_tour(self, cities):
        # Assign cities to their closest neurons
        city_indices = []
        for city in cities:
            distances = np.linalg.norm(self.neurons - city, axis=1)
            winner_idx = np.argmin(distances)
            city_indices.append((city, winner_idx))
        
        # Sort cities based on their neuron index
        city_indices.sort(key=lambda x: x[1])
        return [x[0] for x in city_indices]

# Generate random cities
def generate_cities(num_cities):
    return np.random.rand(num_cities, 2)

# Plot the results
def plot_results(cities, neurons, tour):
    plt.figure(figsize=(10, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', label="Cities")
    
    # Close the neuron path
    closed_neurons = np.vstack((neurons, neurons[0]))
    plt.plot(closed_neurons[:, 0], closed_neurons[:, 1], 'b-', label="Neuron Path")
    
    # Close the TSP tour path
    closed_tour = np.vstack((tour, tour[0]))
    plt.plot(closed_tour[:, 0], closed_tour[:, 1], 'g--', label="TSP Tour")
    
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, f"{i}")
    
    plt.legend()
    plt.title("Travelling Salesman Problem - SOM Solution")
    plt.show()

# Main Function
if __name__ == "__main__":
    # Get input from the user
    num_cities = int(input("Enter the number of cities: "))
    num_neurons = int(input("Enter the number of neurons: "))
    learning_rate = float(input("Enter the initial learning rate (e.g., 0.8): "))
    radius = int(input("Enter the initial radius (e.g., 5): "))
    num_iterations = int(input("Enter the number of iterations: "))

    # Generate random cities
    cities = generate_cities(num_cities)

    # Initialize and train SOM
    som_tsp = SOM_TSP(num_cities, num_neurons, learning_rate, radius, num_iterations)
    som_tsp.train(cities)

    # Get the optimized tour
    tour = som_tsp.get_tour(cities)

    # Plot the results
    plot_results(cities, som_tsp.neurons, np.array(tour))
