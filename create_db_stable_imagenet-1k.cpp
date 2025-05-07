/*
 * Dlib ImageNet Dataset Preprocessor
 *
 * This program processes classified images from the "Stable ImageNet-1K" database
 * to create ready-to-use datasets for the Dlib image processing library.
 *
 * Features:
 * - Loads images from class directories
 * - Resizes images to specified dimensions
 * - Saves processed dataset to a binary file
 * - Supports train/test splitting
 *
 * Author: Cydral
 * Date: 07/05/2025
 * Version: 1.0
 */

#include <dlib/dir_nav.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/matrix.h>
#include <dlib/gui_widgets.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <csignal>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;

 // Define a cross-platform signal handling system
namespace {
    std::atomic<bool> g_terminate_flag(false);

#ifdef _WIN32
    // Windows-specific handler
    BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
        if (ctrl_type == CTRL_C_EVENT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up and closing the program..." << endl;
            return TRUE;
        }
        return FALSE;
    }
#else
    // Unix/Linux/macOS handler
    void signal_handler(int signal) {
        if (signal == SIGINT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up and closing the program..." << endl;
        }
    }
#endif

    // Setup the interrupt handler based on platform
    void setup_interrupt_handler() {
#ifdef _WIN32
        if (!SetConsoleCtrlHandler(console_ctrl_handler, TRUE)) {
            cerr << "ERROR: Could not set control handler" << endl;
        }
#else
        struct sigaction sa;
        sa.sa_handler = signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);
#endif
    }
}

namespace dlib
{
    /**
     * Structure containing information about an ImageNet image
     */
    struct imagenet_info
    {
        std::string filename;       // Full path to the image file
        std::string label;         // Textual label (class description)
        unsigned long numeric_label; // Numeric label (class index)
    };

    /**
     * Structure representing an ImageNet dataset
     */
    struct imagenet_dataset
    {
        std::vector<matrix<rgb_pixel>> images;  // Vector of image matrices
        std::vector<std::string> labels;        // Vector of textual labels
        std::vector<unsigned long> numeric_labels; // Vector of numeric labels
    };

    /**
     * Extracts the class description from directory name
     * Directory names are expected in format "nXXXXXX_description"
     *
     * @param dir_name The directory name to process
     * @return The description part of the directory name
     */
    std::string extract_desc_class(const std::string& dir_name)
    {
        size_t underscore_pos = dir_name.find('_');
        return dir_name.substr(underscore_pos + 1);
    }

    /**
     * Scans an image directory and creates a list of ImageNet images with their metadata
     *
     * @param images_folder Root directory containing class subdirectories
     * @return Vector of imagenet_info structures for all found images
     */
    std::vector<imagenet_info> get_imagenet_listing(
        const std::string& images_folder
    )
    {
        std::vector<imagenet_info> results;
        imagenet_info temp;
        temp.numeric_label = 0;

        // Get all subdirectories (each represents a class)
        auto subdirs = directory(images_folder).get_dirs();

        // Sort subdirectories to assign numeric labels in consistent order
        std::sort(subdirs.begin(), subdirs.end(),
            [](const directory& a, const directory& b) {
                return a.name() < b.name();
            });

        // Process each class directory
        for (auto subdir : subdirs)
        {
            temp.label = extract_desc_class(subdir.name());

            // Process each image in the directory
            for (auto image_file : subdir.get_files())
            {
                // Only process JPG files
                if (image_file.name().size() > 4 &&
                    tolower(image_file.name().substr(image_file.name().size() - 4)) == ".jpg")
                {
                    temp.filename = image_file;
                    results.push_back(temp);
                }
            }
            ++temp.numeric_label;
        }
        return results;
    }

    /**
     * Loads an image from disk and resizes it to specified dimensions
     *
     * @param filename Path to the image file
     * @param rows Desired height of the image
     * @param cols Desired width of the image
     * @return Matrix containing the processed image
     */
    matrix<rgb_pixel> load_and_resize_image(
        const std::string& filename,
        long rows,
        long cols
    )
    {
        matrix<rgb_pixel> img;
        load_image(img, filename);

        // Resize image if dimensions don't match
        if (img.nr() != rows || img.nc() != cols)
        {
            matrix<rgb_pixel> resized(rows, cols);
            resize_image(img, resized, interpolate_bilinear());
            assign_image(img, resized);
        }

        return img;
    }

    /**
     * Creates an ImageNet dataset from a directory of images
     *
     * @param images_folder Root directory containing class subdirectories
     * @param output_file Path to save the processed dataset
     * @param resize_rows Height for resizing images (default 224)
     * @param resize_cols Width for resizing images (default 224)
     */
    void create_imagenet_dataset(
        const std::string& images_folder,
        const std::string& output_file,
        long resize_rows = 224,
        long resize_cols = 224
    )
    {
        std::cout << "Scanning image directory..." << std::endl;
        auto image_listing = get_imagenet_listing(images_folder);
        std::cout << "Total images found: " << image_listing.size() << std::endl;

        if (image_listing.empty())
            throw dlib::error("No images found in directory: " + images_folder);

        imagenet_dataset dataset;

        std::cout << "Loading and processing images..." << std::endl;
        dataset.images.reserve(image_listing.size());
        dataset.labels.reserve(image_listing.size());
        dataset.numeric_labels.reserve(image_listing.size());

        for (size_t i = 0; i < image_listing.size() && !g_terminate_flag.load(); ++i)
        {
            try
            {
                const auto& info = image_listing[i];
                matrix<rgb_pixel> img = load_and_resize_image(info.filename, resize_rows, resize_cols);
                dataset.images.push_back(std::move(img));
                dataset.labels.push_back(info.label);
                dataset.numeric_labels.push_back(info.numeric_label);

                // Print progress every 1000 images
                if ((i + 1) % 1000 == 0 || i == image_listing.size() - 1)
                {
                    std::cout << "Progress: " << (i + 1) << "/" << image_listing.size()
                        << " images processed" << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error processing image " << image_listing[i].filename
                    << ": " << e.what() << std::endl;
            }
        }

        std::cout << "Saving dataset to: " << output_file << std::endl;
        serialize(output_file) << dataset.images << dataset.labels << dataset.numeric_labels;
        std::cout << "Dataset saved successfully!" << std::endl;
    }

    /**
     * Loads a preprocessed ImageNet dataset and splits it into training and testing sets
     *
     * @param dataset_file Path to the saved dataset file
     * @param training_images Output vector for training images
     * @param training_labels Output vector for training labels
     * @param testing_images Output vector for testing images
     * @param testing_labels Output vector for testing labels
     * @param test_fraction Fraction of data to use for testing (default 0.05)
     */
    void load_stable_imagenet_1k(
        const std::string& dataset_file,
        std::vector<matrix<rgb_pixel>>& training_images,
        std::vector<unsigned long>& training_labels,
        std::vector<matrix<rgb_pixel>>& testing_images,
        std::vector<unsigned long>& testing_labels,
        double test_fraction = 0.05
    )
    {
        imagenet_dataset dataset;
        deserialize(dataset_file) >> dataset.images >> dataset.labels >> dataset.numeric_labels;

        // Create indices for shuffling
        std::vector<size_t> indices(dataset.images.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

        // Shuffle indices for random train/test split
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        size_t split_point = static_cast<size_t>(dataset.images.size() * (1.0 - test_fraction));

        // Reserve space for efficiency
        training_images.clear();
        training_labels.clear();
        testing_images.clear();
        testing_labels.clear();
        training_images.reserve(split_point);
        training_labels.reserve(split_point);
        testing_images.reserve(dataset.images.size() - split_point);
        testing_labels.reserve(dataset.images.size() - split_point);

        // Split into training and testing sets
        for (size_t i = 0; i < indices.size(); ++i)
        {
            size_t idx = indices[i];

            if (i < split_point)
            {
                training_images.push_back(dataset.images[idx]);
                training_labels.push_back(dataset.numeric_labels[idx]);
            }
            else
            {
                testing_images.push_back(dataset.images[idx]);
                testing_labels.push_back(dataset.numeric_labels[idx]);
            }
        }
    }
}

/**
 * Main program for creating an ImageNet dataset
 */
int main(int argc, char** argv)
{
    try
    {
        // Setup interrupt handling for clean termination
        setup_interrupt_handler();

        if (argc != 4)
        {
            std::cout << "Usage: " << argv[0] << " <image_directory> <output_file> <image_size>" << std::endl;
            std::cout << "Example: " << argv[0] << " imagenet_train imagenet.dat 224" << std::endl;
            return 1;
        }

        std::string image_directory(argv[1]);
        std::string output_file(argv[2]);
        long image_size = std::stol(argv[3]);

        std::cout << "Creating ImageNet dataset with parameters:" << std::endl;
        std::cout << "  Image directory: " << image_directory << std::endl;
        std::cout << "  Output file: " << output_file << std::endl;
        std::cout << "  Image size: " << image_size << "x" << image_size << std::endl;

        // Create the dataset
        dlib::create_imagenet_dataset(image_directory, output_file, image_size, image_size);

        // Now load and evaluate the dataset
        std::vector<dlib::matrix<dlib::rgb_pixel>> training_images, testing_images;
        std::vector<unsigned long> training_labels, testing_labels;

        // Now load with train/test split
        dlib::load_stable_imagenet_1k(output_file, training_images, training_labels,
            testing_images, testing_labels);

        // Create a window for displaying images
        dlib::image_window win;

        // Display training set info
        std::cout << "\nTraining set (" << training_images.size() << " images):" << std::endl;
        size_t num_to_show = std::min((size_t)3, training_images.size());

        for (size_t i = 0; i < num_to_show; ++i)
        {
            std::cout << "  Image " << i + 1 << " - Label: " << training_labels[i] << std::endl;

            // Display the image
            win.set_image(training_images[i]);
            win.set_title("Training Image #" + std::to_string(i + 1));
            std::cout << "    Press enter to continue..." << std::endl;
            std::cin.ignore();
        }

        // Display testing set info
        std::cout << "\nTesting set (" << testing_images.size() << " images):" << std::endl;
        num_to_show = std::min((size_t)3, testing_images.size());

        for (size_t i = 0; i < num_to_show; ++i)
        {
            std::cout << "  Image " << i + 1 << " - Label: " << testing_labels[i] << std::endl;

            // Display the image
            win.set_image(testing_images[i]);
            win.set_title("Testing Image #" + std::to_string(i + 1));
            std::cout << "    Press enter to continue..." << std::endl;
            std::cin.ignore();
        }

        std::cout << "\nDataset evaluation complete." << std::endl;
        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}