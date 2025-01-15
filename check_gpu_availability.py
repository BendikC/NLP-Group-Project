import torch
import faiss

def check_faiss_gpu():
    try:
        # Check if FAISS GPU is available
        gpu_count = faiss.get_num_gpus()
        print(f"Number of FAISS GPUs available: {gpu_count}")
        
        
        if gpu_count > 0:
            # Get information about each GPU
            for i in range(gpu_count):
                res = faiss.StandardGpuResources()
                print(f"FAISS GPU {i} initialized successfully")
                
            return True, list(range(gpu_count))
        else:
            print("No FAISS GPUs available")
            return False, []
            
    except Exception as e:
        print(f"Error checking FAISS GPU availability: {e}")
        return False, []

def check_torch_gpu():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of PyTorch CUDA GPUs available: {gpu_count}")
        
        # Print details for each GPU
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        return True, list(range(gpu_count))
    else:
        print("No PyTorch CUDA GPUs available")
        return False, []
    
    

def check_faiss_available():
    try:
        # Create a small index as a test
        d = 64  # dimension
        index = faiss.IndexFlatL2(d)
        return True
    except Exception as e:
        print(f"Error initializing FAISS: {e}")
        return False
    
def check_fais_and_cuda_version():
    print("FAISS version:", faiss.__version__)
    print("CUDA version:", torch.version.cuda)
# Add to main:
if __name__ == "__main__":
    check_fais_and_cuda_version()
    faiss_available, faiss_gpus = check_faiss_gpu()
    torch_available, torch_gpus = check_torch_gpu()
    can_use_faiss = check_faiss_available()
    print(f"\nCan use FAISS (CPU or GPU): {can_use_faiss}")
