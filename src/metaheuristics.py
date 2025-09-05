# metaheuristics.py
import random
import image_processing
import evaluation
import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=50, max_gens=100, mutation_rate=0.1, crossover_rate=0.8, pipeline_min_len=4, pipeline_max_len=10, tournament_size=5):
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.pipeline_min_len = pipeline_min_len
        self.pipeline_max_len = pipeline_max_len
        self.tournament_size = tournament_size

        self.operations = [
            image_processing.gamma_correction,
            image_processing.gaussian_blur,
            image_processing.unsharp_masking,
            image_processing.histogram_equalization,
            image_processing.contrast_stretching
        ]
        
        self.param_ranges = {
            'gamma': (0.5, 2.0),
            'kernel_size': [3, 5, 7, 9],
            'strength': (0.5, 2.0)
        }

    def _create_random_operation(self):
        op_func = random.choice(self.operations)
        params = {}
        if op_func == image_processing.gamma_correction:
            params['gamma'] = random.uniform(*self.param_ranges['gamma'])
        elif op_func == image_processing.gaussian_blur:
            params['kernel_size'] = (random.choice(self.param_ranges['kernel_size']), random.choice(self.param_ranges['kernel_size']))
        elif op_func == image_processing.unsharp_masking:
            params['kernel_size'] = (random.choice(self.param_ranges['kernel_size']), random.choice(self.param_ranges['kernel_size']))
            params['strength'] = random.uniform(*self.param_ranges['strength'])
        return {'function': op_func, 'params': params}

    def _create_random_pipeline(self):
        pipeline_len = random.randint(self.pipeline_min_len, self.pipeline_max_len)
        return [self._create_random_operation() for _ in range(pipeline_len)]

    def _initialize_population(self):
        return [self._create_random_pipeline() for _ in range(self.pop_size)]

    def _apply_pipeline(self, image, pipeline):
        processed_image = image.copy()
        for op in pipeline:
            processed_image = op['function'](processed_image, **op['params'])
        return processed_image

    def _calculate_fitness(self, pipeline, image, ground_truth):
        processed_image = self._apply_pipeline(image, pipeline)
        return evaluation.calculate_ssim(processed_image, ground_truth)

    def _selection(self, population, fitness_scores):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate and min(len(parent1), len(parent2)) > 1:
            point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def _mutate(self, pipeline):
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'change_params', 'swap'])

            if mutation_type == 'add' and len(pipeline) < self.pipeline_max_len:
                pipeline.insert(random.randint(0, len(pipeline)), self._create_random_operation())

            elif mutation_type == 'remove' and len(pipeline) > self.pipeline_min_len:
                pipeline.pop(random.randint(0, len(pipeline) - 1))

            elif mutation_type == 'change_params' and len(pipeline) > 0:
                op_to_mutate = random.choice(pipeline)
                new_op = self._create_random_operation()
                op_to_mutate['function'] = new_op['function']
                op_to_mutate['params'] = new_op['params']

            elif mutation_type == 'swap' and len(pipeline) > 1:
                idx1, idx2 = random.sample(range(len(pipeline)), 2)
                pipeline[idx1], pipeline[idx2] = pipeline[idx2], pipeline[idx1]
        return pipeline

    def run(self, train_images, train_gt):
        print("  Running GA...")
        population = self._initialize_population()
        best_pipeline = None
        best_fitness = -1

        for gen in range(self.max_gens):
            print(f"    Generation {gen+1}/{self.max_gens}")
            
            fitness_scores = [np.mean([self._calculate_fitness(p, train_images[j], train_gt[j]) for j in range(len(train_images))]) for p in population]

            best_gen_fitness = max(fitness_scores)
            if best_gen_fitness > best_fitness:
                best_fitness = best_gen_fitness
                best_pipeline = population[np.argmax(fitness_scores)]

            print(f"      Best fitness in generation {gen+1}: {best_gen_fitness:.4f}")

            selected_population = self._selection(population, fitness_scores)
            
            next_population = []
            if len(selected_population) % 2 != 0:
                selected_population.pop()

            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                next_population.extend([self._mutate(child1), self._mutate(child2)])
            
            if best_pipeline is not None:
              next_population[0] = best_pipeline
            population = next_population

        print("  GA finished.")
        return best_pipeline

class VariableNeighbourhoodSearch(GeneticAlgorithm):
    def __init__(self, max_iter=100, k_max=4, local_search_iter=10, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.k_max = k_max
        self.local_search_iter = local_search_iter

    def _shake(self, pipeline, k):
        temp_pipeline = pipeline.copy()
        for _ in range(k):
            temp_pipeline = self._mutate(temp_pipeline)
        return temp_pipeline

    def _local_search(self, pipeline, train_images, train_gt):
        best_pipeline = pipeline
        best_fitness = np.mean([self._calculate_fitness(pipeline, img, gt) for img, gt in zip(train_images, train_gt)])

        for _ in range(self.local_search_iter):
            neighbour = self._mutate(best_pipeline.copy())
            neighbour_fitness = np.mean([self._calculate_fitness(neighbour, img, gt) for img, gt in zip(train_images, train_gt)])

            if neighbour_fitness > best_fitness:
                best_pipeline = neighbour
                best_fitness = neighbour_fitness
        
        return best_pipeline

    def run(self, train_images, train_gt):
        print("  Running VNS...")
        
        best_solution = self._create_random_pipeline()
        best_fitness = np.mean([self._calculate_fitness(best_solution, img, gt) for img, gt in zip(train_images, train_gt)])

        iter_count = 0
        while iter_count < self.max_iter:
            print(f"    Iteration {iter_count+1}/{self.max_iter}")
            k = 1
            while k <= self.k_max:
                shaken_solution = self._shake(best_solution, k)
                local_optimum = self._local_search(shaken_solution, train_images, train_gt)
                local_optimum_fitness = np.mean([self._calculate_fitness(local_optimum, img, gt) for img, gt in zip(train_images, train_gt)])

                if local_optimum_fitness > best_fitness:
                    best_solution = local_optimum
                    best_fitness = local_optimum_fitness
                    print(f"      New best fitness: {best_fitness:.4f} (k={k})")
                    k = 1
                else:
                    k += 1
            iter_count += 1

        print("  VNS finished.")
        return best_solution