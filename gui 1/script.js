const API_BASE_URL = 'http://localhost:9600'; // Assuming lollms_server runs on this port

document.addEventListener('DOMContentLoaded', () => {
    const apiKeyInput = document.getElementById('api-key');
    const loginBtn = document.getElementById('login-btn');
    const authSection = document.getElementById('auth-section');
    const appSection = document.getElementById('app-section');
    const statusMessage = document.getElementById('status-message');

    const bindingSelect = document.getElementById('binding-select');
    const modelSelect = document.getElementById('model-select');
    const personalitySelect = document.getElementById('personality-select');

    const promptInput = document.getElementById('prompt-input');
    const generationTypeSelect = document.getElementById('generation-type');
    const fileUploadGroup = document.getElementById('file-upload-group');
    const fileInput = document.getElementById('file-input');
    const generateBtn = document.getElementById('generate-btn');
    const outputDisplay = document.getElementById('output-display');
    const mediaOutput = document.getElementById('media-output');

    let currentApiKey = '';

    // --- Utility Functions ---
    function showStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? '#FF6347' : '#FFD700'; // Tomato for error, Gold for status
    }

    function clearStatus() {
        statusMessage.textContent = '';
    }

    async function fetchData(endpoint, method = 'GET', body = null) {
        clearStatus();
        try {
            const headers = {
                'Content-Type': 'application/json',
            };
            if (currentApiKey) {
                headers['x-api-key'] = currentApiKey;
            }

            const options = {
                method: method,
                headers: headers,
            };
            if (body) {
                options.body = JSON.stringify(body);
            }

            const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            showStatus(`Error: ${error.message}`, true);
            console.error('Fetch error:', error);
            throw error; // Re-throw to be caught by calling function
        }
    }

    // --- Authentication ---
    loginBtn.addEventListener('click', async () => {
        const apiKey = apiKeyInput.value.trim();
        if (!apiKey) {
            showStatus('Please enter an API Key.', true);
            return;
        }

        // For local UI, we assume a successful login if an API key is provided.
        // In a real scenario, you'd send a request to a /login or /validate endpoint.
        currentApiKey = apiKey;
        showStatus('Attempting to log in...');

        // Try to fetch bindings to validate the API key
        try {
            await fetchBindings();
            authSection.style.display = 'none';
            appSection.style.display = 'flex'; // Use flex to maintain column layout
            showStatus('Login successful. Welcome!', false);
        } catch (error) {
            showStatus('Login failed. Invalid API Key or server not reachable.', true);
            currentApiKey = ''; // Clear API key on failure
        }
    });

    // --- Dynamic Dropdown Population ---
    async function fetchBindings() {
        bindingSelect.innerHTML = '<option value="">Loading Bindings...</option>';
        try {
            const data = await fetchData('/list_bindings');
            bindingSelect.innerHTML = '<option value="">-- Select a Binding --</option>';
            data.bindings.forEach(binding => {
                const option = document.createElement('option');
                option.value = binding.name;
                option.textContent = binding.name;
                bindingSelect.appendChild(option);
            });
            if (data.bindings.length > 0) {
                bindingSelect.value = data.bindings[0].name; // Select first by default
                fetchModels(data.bindings[0].name); // Fetch models for the first binding
            }
        } catch (error) {
            showStatus('Failed to load bindings.', true);
        }
    }

    async function fetchModels(bindingName) {
        modelSelect.innerHTML = '<option value="">Loading Models...</option>';
        if (!bindingName) {
            modelSelect.innerHTML = '<option value="">-- Select a Binding First --</option>';
            return;
        }
        try {
            const data = await fetchData(`/list_models?binding_name=${bindingName}`);
            modelSelect.innerHTML = '<option value="">-- Select a Model --</option>';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;
                modelSelect.appendChild(option);
            });
            if (data.models.length > 0) {
                modelSelect.value = data.models[0].name; // Select first by default
            }
        } catch (error) {
            showStatus(`Failed to load models for ${bindingName}.`, true);
        }
    }

    async function fetchPersonalities() {
        personalitySelect.innerHTML = '<option value="">Loading Personalities...</option>';
        try {
            const data = await fetchData('/list_personalities');
            personalitySelect.innerHTML = '<option value="">-- Select a Personality (Optional) --</option>';
            data.personalities.forEach(personality => {
                const option = document.createElement('option');
                option.value = personality.name;
                option.textContent = personality.name;
                personalitySelect.appendChild(option);
            });
        } catch (error) {
            showStatus('Failed to load personalities.', true);
        }
    }

    bindingSelect.addEventListener('change', (event) => {
        fetchModels(event.target.value);
    });

    // --- Generation Logic ---
    generationTypeSelect.addEventListener('change', (event) => {
        if (event.target.value !== 'text') {
            fileUploadGroup.style.display = 'block';
        } else {
            fileUploadGroup.style.display = 'none';
        }
    });

    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        const bindingName = bindingSelect.value;
        const modelName = modelSelect.value;
        const personalityName = personalitySelect.value;
        const generationType = generationTypeSelect.value;
        const file = fileInput.files[0];

        if (!prompt && !file) {
            showStatus('Please enter a prompt or upload a file.', true);
            return;
        }
        if (!bindingName || !modelName) {
            showStatus('Please select a binding and a model.', true);
            return;
        }

        outputDisplay.textContent = 'Generating...';
        mediaOutput.innerHTML = '';
        showStatus('Sending generation request...');

        let requestBody = {
            prompt: prompt,
            binding_name: bindingName,
            model_name: modelName,
            generation_type: generationType,
        };

        if (personalityName) {
            requestBody.personality_name = personalityName;
        }

        if (file) {
            const reader = new FileReader();
            reader.onload = async (e) => {
                const base64Data = e.target.result.split(',')[1]; // Get base64 string after comma
                requestBody.input_data = [{
                    type: file.type.split('/')[0], // 'image', 'audio', 'video'
                    role: 'input_file', // A generic role for now
                    data: base64Data,
                    mime_type: file.type
                }];
                await sendGenerationRequest(requestBody);
            };
            reader.onerror = (error) => {
                showStatus('Error reading file.', true);
                console.error('File reading error:', error);
            };
            reader.readAsDataURL(file);
        } else {
            await sendGenerationRequest(requestBody);
        }
    });

    async function sendGenerationRequest(requestBody) {
        try {
            const data = await fetchData('/generate', 'POST', requestBody);
            outputDisplay.textContent = data.text || '';

            if (data.image_base64) {
                const img = document.createElement('img');
                img.src = `data:${data.image_mime_type || 'image/png'};base64,${data.image_base64}`;
                mediaOutput.appendChild(img);
            } else if (data.audio_base64) {
                const audio = document.createElement('audio');
                audio.controls = true;
                audio.src = `data:${data.audio_mime_type || 'audio/wav'};base64,${data.audio_base64}`;
                mediaOutput.appendChild(audio);
            } else if (data.video_base64) {
                const video = document.createElement('video');
                video.controls = true;
                video.src = `data:${data.video_mime_type || 'video/mp4'};base64,${data.video_base64}`;
                mediaOutput.appendChild(video);
            }
            showStatus('Generation complete!', false);
        } catch (error) {
            outputDisplay.textContent = 'Error during generation.';
            // Status message already handled by fetchData
        }
    }

    // Initial setup (hide app section, show auth)
    authSection.style.display = 'flex'; // Use flex to maintain column layout
    appSection.style.display = 'none';
});


