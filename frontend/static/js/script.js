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
            fileName.textContent = this.files[0].name;
            
            // Create FormData and send the file
            const formData = new FormData();
            formData.append('file', this.files[0]);
            
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
                
                // Enable buttons after successful upload
                originalBtn.classList.remove('disabled');
                preprocessBtn.classList.remove('disabled');
                augmentBtn.classList.remove('disabled');
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
            
            if (data.type === 'audio') {
                const timestamp = new Date().getTime();
                originalOutput.innerHTML = `
                    <audio controls>
                        <source src="${data.audioPath}?t=${timestamp}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                `;
            } else {
                originalOutput.textContent = data.output;
            }
        } catch (error) {
            console.error('Error:', error);
            originalOutput.textContent = `Error: ${error.message}\nPlease make sure the backend server is running on port 8000`;
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
            
            if (data.type === 'audio') {
                const timestamp = new Date().getTime();
                preprocessOutput.innerHTML = `
                    <audio controls>
                        <source src="${data.audioPath}?t=${timestamp}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
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
            
            const response = await fetch('http://localhost:8000/api/augment', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.type === 'audio') {
                const timestamp = new Date().getTime();
                augmentOutput.innerHTML = `
                    <audio controls>
                        <source src="${data.audioPath}?t=${timestamp}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
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