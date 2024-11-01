from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import traceback

import text_preprocess_and_augmentation_main
from text_preprocess_and_augmentation_main import show_original_data,show_lowercase_data,\
show_after_remove_stop_words_data,show_after_synonym_replacement_data,show_after_random_insertion_data

import logging

# Add at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("../frontend/templates/index.html")

@app.post("/api/preprocess")
async def preprocess_data(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            lowercase_result = show_lowercase_data(temp_file_path)
            stopwords_result = show_after_remove_stop_words_data(temp_file_path)
            
            return {
                "lowercase_data": lowercase_result,
                "after_remove_stop_words_data": stopwords_result
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment")
async def augment_data(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            synonym_result = show_after_synonym_replacement_data(temp_file_path)
            insertion_result = show_after_random_insertion_data(temp_file_path)
            
            return {
                "synonym_replacement_data": synonym_result,
                "random_insertion_data": insertion_result
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/original")
async def original_data(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Create a temporary file with a unique name
        temp_file_path = f"temp_{file.filename}"
        
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            result = show_original_data(temp_file_path)
            
            return {"output": result}
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())  # Log the full error traceback
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 