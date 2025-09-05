
# main.py
import os
import cv2
import numpy as np
import image_processing
import evaluation
from metaheuristics import GeneticAlgorithm, VariableNeighbourhoodSearch

def load_images(path):
    """Loads distorted and ground truth images from a given path."""
    distorted_images = []
    gt_images = []
    
    # Get absolute path
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))

    for file_name in sorted(os.listdir(path)):
        if file_name.endswith('_gt.jpg'):
            gt_images.append(cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE))
        elif file_name.endswith('.jpg'):
            distorted_images.append(cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE))
            
    print(f"Loaded {len(distorted_images)} distorted and {len(gt_images)} ground truth images from {path}")
    
    return distorted_images, gt_images

def run_ga_pipeline(train_distorted, train_gt, test_distorted, test_gt, output_path):
    """Runs the Genetic Algorithm pipeline."""
    print("\n----- Running Genetic Algorithm pipeline... ----- ")
    ga = GeneticAlgorithm(pop_size=20, max_gens=50, mutation_rate=0.2, crossover_rate=0.8)
    best_pipeline = ga.run(train_distorted, train_gt)

    print("\nGA finished. Applying best pipeline to test images...")
    
    ga_output_path = os.path.join(output_path, 'ga')
    os.makedirs(ga_output_path, exist_ok=True)

    results = []
    for i, test_image in enumerate(test_distorted):
        enhanced_image = ga._apply_pipeline(test_image, best_pipeline)
        cv2.imwrite(os.path.join(ga_output_path, f'test_image_{i+1}_enhanced.jpg'), enhanced_image)
        mse = evaluation.calculate_mse(enhanced_image, test_gt[i])
        psnr = evaluation.calculate_psnr(enhanced_image, test_gt[i])
        ssim = evaluation.calculate_ssim(enhanced_image, test_gt[i])
        results.append((mse, psnr, ssim))

    print("\nGA Test Results:")
    for i, res in enumerate(results):
        print(f"  Test Image {i+1}: MSE={res[0]:.2f}, PSNR={res[1]:.2f}, SSIM={res[2]:.4f}")
    
    return results

def run_vns_pipeline(train_distorted, train_gt, test_distorted, test_gt, output_path):
    """Runs the Variable Neighbourhood Search pipeline."""
    print("\n----- Running Variable Neighbourhood Search pipeline... ----- ")
    vns = VariableNeighbourhoodSearch(max_iter=50, k_max=4, local_search_iter=10)
    best_pipeline = vns.run(train_distorted, train_gt)

    print("\nVNS finished. Applying best pipeline to test images...")

    vns_output_path = os.path.join(output_path, 'vns')
    os.makedirs(vns_output_path, exist_ok=True)

    results = []
    for i, test_image in enumerate(test_distorted):
        enhanced_image = vns._apply_pipeline(test_image, best_pipeline)
        cv2.imwrite(os.path.join(vns_output_path, f'test_image_{i+1}_enhanced.jpg'), enhanced_image)
        mse = evaluation.calculate_mse(enhanced_image, test_gt[i])
        psnr = evaluation.calculate_psnr(enhanced_image, test_gt[i])
        ssim = evaluation.calculate_ssim(enhanced_image, test_gt[i])
        results.append((mse, psnr, ssim))

    print("\nVNS Test Results:")
    for i, res in enumerate(results):
        print(f"  Test Image {i+1}: MSE={res[0]:.2f}, PSNR={res[1]:.2f}, SSIM={res[2]:.4f}")

    return results

def main():
    """Main function to run the image enhancement experiment."""
    
    training_path = "../input/training"
    test_path = "../input/test"
    output_path = "output"
    
    train_distorted, train_gt = load_images(training_path)
    test_distorted, test_gt = load_images(test_path)
    
    ga_results = run_ga_pipeline(train_distorted, train_gt, test_distorted, test_gt, output_path)
    vns_results = run_vns_pipeline(train_distorted, train_gt, test_distorted, test_gt, output_path)
    
    print("\n----- Final Results Comparison ----- ")
    print("Image | GA (MSE, PSNR, SSIM)      | VNS (MSE, PSNR, SSIM)")
    print("---------------------------------------------------------------------")
    for i in range(len(test_distorted)):
        ga_res = ga_results[i]
        vns_res = vns_results[i]
        print(f"  {i+1}   | {ga_res[0]:.2f}, {ga_res[1]:.2f}, {ga_res[2]:.4f} | {vns_res[0]:.2f}, {vns_res[1]:.2f}, {vns_res[2]:.4f}")

    print("\nExperiment finished.")

if __name__ == "__main__":
    main()
