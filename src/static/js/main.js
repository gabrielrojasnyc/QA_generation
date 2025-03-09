// Main JavaScript for PDF Q&A Generator

// Store session ID
let sessionId = null;

// Check for saved API keys on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check for saved API keys in localStorage
    const openaiKey = localStorage.getItem('openaiApiKey');
    const anthropicKey = localStorage.getItem('anthropicApiKey');
    
    if (openaiKey) {
        document.getElementById('openaiKey').value = openaiKey;
    }
    if (anthropicKey) {
        document.getElementById('anthropicKey').value = anthropicKey;
    }
});

// Save API keys
document.getElementById('saveKeys').addEventListener('click', function() {
    const openaiKey = document.getElementById('openaiKey').value.trim();
    const anthropicKey = document.getElementById('anthropicKey').value.trim();
    
    if (openaiKey) {
        localStorage.setItem('openaiApiKey', openaiKey);
    }
    if (anthropicKey) {
        localStorage.setItem('anthropicApiKey', anthropicKey);
    }
    
    alert('API keys saved locally in your browser.');
});

// Upload form submission
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('pdfFile');
    const modelSelect = document.getElementById('modelSelect');
    
    if (!fileInput.files.length) {
        alert('Please select a PDF file.');
        return;
    }
    
    // Set API keys as environment variables
    const openaiKey = document.getElementById('openaiKey').value.trim();
    const anthropicKey = document.getElementById('anthropicKey').value.trim();
    
    // Check if keys are available based on selected model
    const modelId = modelSelect.value;
    if ((modelId === "1" || modelId === "2") && !openaiKey) {
        alert('OpenAI API key is required for the selected model.');
        return;
    }
    if ((modelId === "3" || modelId === "4") && !anthropicKey) {
        alert('Anthropic API key is required for the selected model.');
        return;
    }
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', modelSelect.value);
    
    // Show loading indicator
    document.getElementById('loadingContainer').style.display = 'block';
    document.getElementById('resultsContainer').style.display = 'none';
    
    // Send request to process PDF
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            sessionId = data.session_id;
            displayResults(data.display_data);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the PDF.');
    })
    .finally(() => {
        document.getElementById('loadingContainer').style.display = 'none';
    });
});

// Display results
function displayResults(displayData) {
    const container = document.getElementById('chunksContainer');
    container.innerHTML = '';
    
    displayData.forEach(item => {
        const chunkElement = document.createElement('div');
        chunkElement.className = 'chunk-container';
        
        // Chunk text
        const textElement = document.createElement('div');
        textElement.className = 'mb-3';
        textElement.innerHTML = `<strong>Chunk ${item.chunk_id + 1}:</strong> <span class="text-muted">${item.text}</span>`;
        chunkElement.appendChild(textElement);
        
        // QA pairs for this chunk
        const qaContainer = document.createElement('div');
        qaContainer.className = 'mb-3';
        qaContainer.id = `qa-container-${item.chunk_id}`;
        
        // Add QA pairs
        qaContainer.innerHTML = '<h5>Q&A Pairs:</h5>';
        item.qa_pairs.forEach(qa => {
            const qaElement = document.createElement('div');
            qaElement.className = 'qa-item mb-2';
            qaElement.innerHTML = `
                <div><strong>Q:</strong> ${qa.question}</div>
                <div><strong>A:</strong> ${qa.answer}</div>
            `;
            qaContainer.appendChild(qaElement);
        });
        
        chunkElement.appendChild(qaContainer);
        
        // Regenerate button
        const regenerateBtn = document.createElement('button');
        regenerateBtn.className = 'btn btn-sm btn-outline-primary';
        regenerateBtn.textContent = 'Regenerate Q&A';
        regenerateBtn.dataset.chunkId = item.chunk_id;
        regenerateBtn.addEventListener('click', handleRegenerateClick);
        chunkElement.appendChild(regenerateBtn);
        
        container.appendChild(chunkElement);
    });
    
    document.getElementById('resultsContainer').style.display = 'block';
}

// Handle regenerate button click
function handleRegenerateClick(e) {
    const chunkId = e.target.dataset.chunkId;
    const qaContainer = document.getElementById(`qa-container-${chunkId}`);
    
    // Disable button during processing
    e.target.disabled = true;
    e.target.textContent = 'Regenerating...';
    
    // Add a temporary loading indicator
    qaContainer.innerHTML = '<div class="loader" style="width: 20px; height: 20px;"></div>';
    
    // Send request to regenerate QA
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('chunk_id', chunkId);
    
    fetch('/regenerate_qa', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update QA pairs
            qaContainer.innerHTML = '<h5>Q&A Pairs:</h5>';
            data.qa_pairs.forEach(qa => {
                const qaElement = document.createElement('div');
                qaElement.className = 'qa-item mb-2';
                qaElement.innerHTML = `
                    <div><strong>Q:</strong> ${qa.question}</div>
                    <div><strong>A:</strong> ${qa.answer}</div>
                `;
                qaContainer.appendChild(qaElement);
            });
        } else {
            alert('Error: ' + data.error);
            qaContainer.innerHTML = '<p class="text-danger">Failed to regenerate Q&A pairs.</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while regenerating Q&A pairs.');
        qaContainer.innerHTML = '<p class="text-danger">Failed to regenerate Q&A pairs.</p>';
    })
    .finally(() => {
        // Re-enable button
        e.target.disabled = false;
        e.target.textContent = 'Regenerate Q&A';
    });
}

// Ask question form submission
document.getElementById('questionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const questionInput = document.getElementById('userQuestion');
    const question = questionInput.value.trim();
    
    if (!question) {
        alert('Please enter a question.');
        return;
    }
    
    // Disable form during processing
    const submitBtn = this.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';
    
    // Hide previous answer if any
    document.getElementById('answerContainer').style.display = 'none';
    
    // Send request to ask question
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('question', question);
    
    fetch('/ask_question', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Display answer
            document.getElementById('answerText').textContent = data.answer;
            document.getElementById('answerContainer').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the question.');
    })
    .finally(() => {
        // Re-enable form
        submitBtn.disabled = false;
        submitBtn.textContent = 'Ask';
    });
});

// Export to CSV
document.getElementById('exportCsv').addEventListener('click', function() {
    if (!sessionId) {
        alert('No session data available. Please process a PDF first.');
        return;
    }
    
    // Disable button during processing
    this.disabled = true;
    this.textContent = 'Exporting...';
    
    // Send request to export CSV
    const formData = new FormData();
    formData.append('session_id', sessionId);
    
    fetch('/export_csv', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Create download link
            const a = document.createElement('a');
            a.href = data.csv_url;
            a.download = 'qa_pairs.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while exporting to CSV.');
    })
    .finally(() => {
        // Re-enable button
        this.disabled = false;
        this.textContent = 'Export to CSV';
    });
});