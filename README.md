# Face_Dog_Recognition

#install ncnn with vulkal (run with root)
    
    cd /opt/
    
    git clone https://github.com/Tencent/ncnn.git
    
    cd ncnn
    
    git submodule update --init
    
    mkdir -p build
    
    cd build
    
    cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
    
    make -j$(nproc)

# download model

    mkdir models

    cd models
  
    model detection
        
        gdown https://drive.google.com/uc?id=12v08Z5WWUzCpQIX8ufTTBm2GLUrEAr0N
        
        gdown https://drive.google.com/uc?id=1aKSlv-tFTMgKWeZ_yoH92jO-1os7vJpA
        
   
    model extraction:
        
        gdown https://drive.google.com/uc?id=1bDJd6w4_4VBAlB7QsM3Ua99iPz3qgmkd
        
        gdown https://drive.google.com/uc?id=14A-WKmRqBPqw49APYYdes96oiHroAiGD




 
 
 # install
 
    mkdir build

    cd build

    export ncnn_DIR=YOUR_NCNN_PATH/build/install/lib/cmake/ncnn

    cmake ..

    make

# use
    
    ./main
    
    if use gpu :
        
        modify in src/main.cpp:
            
                FaceDogDetector *fdd = new FaceDogDetector("../weights/facedog.param", "../weights/facedog.bin",
                                                           0.7, 0.5, 640, 640, true);

                FaceDogExtractor *face_dog_extractor = new FaceDogExtractor("../weights/facedog_res18_1.param",
                                                                            "../weights/facedog_res18_1.bin", 112, 112, true);
  

   
    
