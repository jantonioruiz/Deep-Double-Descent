#!/bin/bash

#SBATCH --job-name DDD          # Nombre del proceso en SLURM                    
#SBATCH --gres=gpu:1            # Solicitamos 1 GPU
#SBATCH --mem 20g               # Memoria RAM asignada al proceso: 20 GB
#SBATCH --partition dios        # Partición donde se lanzará el proceso (siempre "dios")
#SBATCH -w atenea               # Nodo específico de la partición (Opciones: "atenea", "hera" y "dionisio")

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/jaruiz/testpy310
export TFHUB_CACHE_DIR=.

# ============================
#    ARGUMENTOS DISPONIBLES
# ============================

# --dataset:            Dataset a usar (Opciones: "MNIST", "CIFAR10" o "CIFAR100")
# --model:              Modelo a utilizar (Opciones: "TwoLayerNN", "DeepNN", "ThreeLayerCNN", "DeepCNN", "PreActResNet", "ResNet18")
# --output_train:       Archivo donde guardar las métricas del entrenamiento
# --output_test:        Archivo donde guardar las métricas de test
# --units:              Número de unidades/filtros para las distintas arquitecturas (en caso de querer un determinado rango usar --units_range UNITS_MIN UNITS_MAX)
# --data_augmentation:  Habilita el aumento de datos (Se aplica RandomCrop(32, 4) y RandomHorizontalFlip())
# --num_train_samples:  Número de ejemplos utilizados para el entrenamiento (por defecto: 4000)
# --num_test_samples:   Número de ejemplos utilizados para evaluación o test (por defecto: 1000)
# --noise:              Nivel de ruido para agregar a las etiquetas (por defecto: 0.1 = 10%)
# --epochs:             Número total de épocas de entrenamiento (por defecto: 1000)
# --batch_size:         Tamaño de batch (por defecto: 128)
# --criterion:          Función de pérdida (por defecto: CrossEntropyLoss)
# --optimizer:          Optimizador (por defecto: Adam)
# --learning_rate:      Tasa de aprendizaje (por defecto: 0.001)

python ../src/main.py --dataset MNIST --model TwoLayerNN --output_train train.txt --output_test test.txt --units 1