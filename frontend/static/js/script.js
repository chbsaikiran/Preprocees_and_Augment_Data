document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const originalBtn = document.getElementById('originalBtn');
    const preprocessBtn = document.getElementById('preprocessBtn');
    const augmentBtn = document.getElementById('augmentBtn');
    const originalOutput = document.querySelector('#originalOutput .output-content');
    const preprocessOutput = document.querySelector('#preprocessOutput .output-content');
    const augmentOutput = document.querySelector('#augmentOutput .output-content');

    // Disable buttons initially
    originalBtn.classList.add('disabled');
    preprocessBtn.classList.add('disabled');
    augmentBtn.classList.add('disabled');

    // File input handler
    fileInput.addEventListener('change', async function(e) {
        if (this.files.length > 0) {
            const file = this.files[0];
            fileName.textContent = file.name;
            
            // Show/hide angle input based on file type
            const imageControls = document.getElementById('imageControls');
            imageControls.style.display = file.type.startsWith('image/') ? 'block' : 'none';
            
            // Enable buttons
            originalBtn.classList.remove('disabled');
            preprocessBtn.classList.remove('disabled');
            augmentBtn.classList.remove('disabled');
            
            // Create FormData and send the file
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('http://localhost:8000/api/save_in_out_file_name/', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('File saved:', data.file_path);
            } catch (error) {
                console.error('Error saving file:', error);
                fileName.textContent = 'Error saving file';
            }
        } else {
            fileName.textContent = '';
            originalBtn.classList.add('disabled');
            preprocessBtn.classList.add('disabled');
            augmentBtn.classList.add('disabled');
        }
    });

    // Original data button handler
    originalBtn.addEventListener('click', async function() {
        if (originalBtn.classList.contains('disabled')) return;
        
        originalOutput.textContent = 'Processing...';
        
        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const response = await fetch('http://localhost:8000/api/original', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.type === 'image') {
                const timestamp = new Date().getTime();
                originalOutput.innerHTML = `
                    <div class="image-container">
                        <img src="http://localhost:8000${data.imagePath}?t=${timestamp}" 
                             alt="Original Image" 
                             class="processed-image">
                    </div>
                `;
            } else if (data.type === 'audio') {
                const timestamp = new Date().getTime();
                originalOutput.innerHTML = `
                    <div class="audio-container">
                        <audio controls>
                            <source src="http://localhost:8000${data.audioPath}?t=${timestamp}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                    <div class="spectrogram-container">
                        <img src="http://localhost:8000${data.spectrogramPath}?t=${timestamp}" 
                             alt="MFCC Spectrogram" 
                             class="spectrogram-image">
                    </div>
                `;
            } else {
                originalOutput.textContent = data.output;
            }
        } catch (error) {
            console.error('Error:', error);
            originalOutput.textContent = `Error: ${error.message}`;
        }
    });

    // Preprocess button handler
    preprocessBtn.addEventListener('click', async function() {
        if (preprocessBtn.classList.contains('disabled')) return;
        
        preprocessOutput.textContent = 'Processing...';
        
        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const response = await fetch('http://localhost:8000/api/preprocess', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.type === 'image') {
                const timestamp = new Date().getTime();
                preprocessOutput.innerHTML = `
                    <div class="image-container">
                        <img src="http://localhost:8000${data.imagePath}?t=${timestamp}" 
                             alt="Processed Image" 
                             class="processed-image">
                    </div>
                `;
            } else if (data.type === 'audio') {
                const timestamp = new Date().getTime();
                preprocessOutput.innerHTML = `
                    <div class="audio-container">
                        <audio controls>
                            <source src="http://localhost:8000${data.audioPath}?t=${timestamp}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                    <div class="spectrogram-container">
                        <img src="http://localhost:8000${data.spectrogramPath}?t=${timestamp}" 
                             alt="MFCC Spectrogram" 
                             class="spectrogram-image">
                    </div>
                `;
            } else {
                preprocessOutput.textContent = 
                    "Lowercase Data:\n" +
                    data.lowercase_data + 
                    "\n\nAfter Remove Stop Words:\n" +
                    data.after_remove_stop_words_data;
            }
        } catch (error) {
            console.error('Error:', error);
            preprocessOutput.textContent = `Error: ${error.message}\nPlease make sure the backend server is running on port 8000`;
        }
    });

    // Augment button handler
    augmentBtn.addEventListener('click', async function() {
        if (augmentBtn.classList.contains('disabled')) return;
        augmentOutput.textContent = 'Processing...';
        
        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // Add rotation angle only for images
            if (fileInput.files[0].type.startsWith('image/')) {
                const angleValue = document.getElementById('rotationAngle').value;
                formData.append('angle', angleValue);
                console.log('Sending angle:', angleValue);
            }
            
            const response = await fetch('http://localhost:8000/api/augment', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.type === 'image') {
                const timestamp = new Date().getTime();
                augmentOutput.innerHTML = `
                    <div class="image-container">
                        <img src="http://localhost:8000${data.imagePath}?t=${timestamp}" 
                             alt="Augmented Image" 
                             class="processed-image">
                    </div>
                `;
            } else if (data.type === 'audio') {
                const timestamp = new Date().getTime();
                augmentOutput.innerHTML = `
                    <div class="audio-container">
                        <audio controls>
                            <source src="http://localhost:8000${data.audioPath}?t=${timestamp}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                    <div class="spectrogram-container">
                        <img src="http://localhost:8000${data.spectrogramPath}?t=${timestamp}" 
                             alt="MFCC Spectrogram" 
                             class="spectrogram-image">
                    </div>
                `;
            } else {
                augmentOutput.textContent = 
                    "Synonym Replacement:\n" +
                    data.synonym_replacement_data +
                    "\n\nRandom Insertion:\n" +
                    data.random_insertion_data;
            }
        } catch (error) {
            console.error('Error:', error);
            augmentOutput.textContent = `Error: ${error.message}\nPlease make sure the backend server is running on port 8000`;
        }
    });
}); 