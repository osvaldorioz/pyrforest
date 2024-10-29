from fastapi import FastAPI
import random_forest_module
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

class Vector(BaseModel):
    vector: List[int]

@app.post("/rforest")
async def forest(nTrees: int,
                 maxDepth: int,
                 maxFeatures: int,
                 X: Matrix, 
                 y: Vector):
    
    start = time.time()
    
    # Crear y entrenar el Random Forest
    forest = random_forest_module.RandomForest(num_trees=nTrees, 
                                               max_depth=maxDepth, 
                                               max_features=maxFeatures)
    forest.fit(X.matrix, y.vector)
    str1 = f"|"

    # Prueba de predicción
    for x in X:
        str1 += f" Input: {x}, Predicted: {forest.predict(x)} | "

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Prediccion": str1
    }
    jj = json.dumps(j1)

    return jj
