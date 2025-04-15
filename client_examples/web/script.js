// client_examples/web/script.js

// --- DOM Elements ---
const serverUrlInput = document.getElementById('server-url');
const apiKeyInput = document.getElementById('api-key');
const connectButton = document.getElementById('connect-button');
const connectStatus = document.getElementById('connect-status');

const generationControls = document.getElementById('generation-controls');
const personalitySelect = document.getElementById('personality-select');
const bindingSelect = document.getElementById('binding-select');
const modelSelect = document.getElementById('model-select');
const refreshModelsButton = document.getElementById('refresh-models-button');
const generationTypeSelect = document.getElementById('generation-type');
const promptInput = document.getElementById('prompt');
const streamCheckbox = document.getElementById('stream-checkbox');
const addParamsButton = document.getElementById('add-params-button');
const paramsContainer = document.getElementById('params-container');
const paramsInput = document.getElementById('parameters');
const addExtraDataButton = document.getElementById('add-extra-data-button');
const extraDataContainer = document.getElementById('extra-data-container');
const extraDataInput = document.getElementById('extra-data');
const generateButton = document.getElementById('generate-button');
const generateStatus = document.getElementById('generate-status');

const resultsOutput = document.getElementById('results-output');
const resultsMetadata = document.getElementById('results-metadata');

// --- Global State ---
let client = null;
let isLoading = false; // Prevent multiple simultaneous requests

// --- Functions ---

function setStatus(element, message, isError = false, isSuccess = false) {
    element.textContent = message;
    element.className = 'status'; // Reset classes
    if (isError) {
        element.classList.add('error');
    } else if (isSuccess) {
        element.classList.add('success');
    }
}

function setLoadingState(loading) {
    isLoading = loading;
    generateButton.disabled = loading;
    connectButton.disabled = loading;
    refreshModelsButton.disabled = loading || !bindingSelect.value;
    generateButton.textContent = loading ? 'Generating...' : 'Generate';
}

function populateSelect(selectElement, items, valueKey, textKey, defaultOptionText) {
    selectElement.innerHTML = `<option value="">-- ${defaultOptionText} --</option>`; // Clear and add default
    if (items && items.length > 0) {
        items.forEach(item => {
            const value = typeof item === 'string' ? item : item[valueKey];
            const text = typeof item === 'string' ? item : item[textKey] || value; // Use value if textKey missing
            const option = document.createElement('option');
            option.value = value;
            option.textContent = text;
            selectElement.appendChild(option);
        });
    }
}

async function loadModelsForBinding() {
    const bindingName = bindingSelect.value;
    modelSelect.innerHTML = '<option value="">-- Loading... --</option>'; // Clear previous
    modelSelect.disabled = true;
    refreshModelsButton.disabled = true;

    if (!client || !bindingName) {
        modelSelect.innerHTML = '<option value="">-- Select Binding First --</option>';
        return;
    }

    setStatus(generateStatus, `Fetching models for ${bindingName}...`);
    try {
        const response = await client.listAvailableModels(bindingName);
        populateSelect(modelSelect, response.models || [], 'name', 'name', 'Use Default');
        setStatus(generateStatus, `Models loaded for ${bindingName}.`, false, true);
        modelSelect.disabled = false;
    } catch (error) {
        setStatus(generateStatus, `Error loading models for ${bindingName}: ${error.message}`, true);
        modelSelect.innerHTML = '<option value="">-- Error Loading --</option>';
    } finally {
         refreshModelsButton.disabled = !bindingSelect.value; // Re-enable if binding still selected
    }
}

async function initializeClientAndLoadData() {
    const baseUrl = serverUrlInput.value.trim();
    const apiKey = apiKeyInput.value.trim();

    if (!baseUrl) {
        setStatus(connectStatus, "Server URL cannot be empty.", true);
        return;
    }
    setStatus(connectStatus, "Connecting and loading data...");
    generationControls.style.display = 'none'; // Hide controls until loaded
    generateButton.disabled = true;
    refreshModelsButton.disabled = true;

    try {
        client = new LollmsClient(baseUrl, apiKey || null); // Pass null if empty

        // Fetch bindings and personalities in parallel
        const [bindingsRes, personalitiesRes] = await Promise.all([
            client.listBindings(),
            client.listPersonalities()
        ]);

        // Populate Bindings
        const bindingInstances = bindingsRes.binding_instances ? Object.keys(bindingsRes.binding_instances) : [];
        populateSelect(bindingSelect, bindingInstances, null, null, 'Use Default');

        // Populate Personalities
        const personalities = personalitiesRes.personalities ? Object.values(personalitiesRes.personalities) : [];
        populateSelect(personalitySelect, personalities, 'name', 'name', 'None');

        setStatus(connectStatus, "Connected and data loaded.", false, true);
        generationControls.style.display = 'block'; // Show controls
        generateButton.disabled = false; // Enable generate button
        bindingSelect.disabled = false;
        personalitySelect.disabled = false;
        modelSelect.disabled = !bindingSelect.value; // Disable models until binding selected
        refreshModelsButton.disabled = !bindingSelect.value;


    } catch (error) {
        setStatus(connectStatus, `Connection failed: ${error.message}`, true);
        client = null; // Reset client on failure
    }
}

function handleGeneration() {
    if (isLoading || !client) return;

    const personality = personalitySelect.value || null; // Send null if empty
    const bindingName = bindingSelect.value || null;
    const modelName = modelSelect.value || null;
    const prompt = promptInput.value.trim();
    const generationType = generationTypeSelect.value;
    const stream = streamCheckbox.checked && generationType === 'ttt'; // Only allow stream for TTT

    if (!prompt) {
        setStatus(generateStatus, "Prompt cannot be empty.", true);
        return;
    }

    setLoadingState(true);
    setStatus(generateStatus, "Generating...");
    resultsOutput.textContent = ''; // Clear previous output
    resultsMetadata.textContent = '';

    const payload = {
        personality: personality,
        prompt: prompt,
        binding_name: bindingName,
        model_name: modelName,
        generation_type: generationType,
        stream: stream
    };

    // Add optional parameters
    if (paramsContainer.style.display !== 'none' && paramsInput.value.trim()) {
        try {
            payload.parameters = JSON.parse(paramsInput.value.trim());
        } catch (e) {
             setStatus(generateStatus, `Invalid JSON in Parameters: ${e.message}`, true);
             setLoadingState(false);
             return;
        }
    }

     // Add optional extra data
    if (extraDataContainer.style.display !== 'none' && extraDataInput.value.trim()) {
        try {
            payload.extra_data = JSON.parse(extraDataInput.value.trim());
        } catch (e) {
             setStatus(generateStatus, `Invalid JSON in Extra Data: ${e.message}`, true);
             setLoadingState(false);
             return;
        }
    }


    // --- Define Callbacks ---
    const streamCb = (chunk) => {
        // console.log("Stream chunk:", chunk);
        if (chunk.type === 'chunk' && chunk.content) {
             // Append text content for TTT stream
             resultsOutput.textContent += chunk.content;
        } else if (chunk.type === 'error'){
             setStatus(generateStatus, `Stream Error: ${chunk.content}`, true);
        }
        // Add handling for 'info' or other types if needed
    };

    const finalCb = (result) => {
        console.log("Final result/chunk:", result);
        setStatus(generateStatus, "Generation complete.", false, true);
         setLoadingState(false);

        // If image generation (TTI), display image
        if (generationType === 'tti' && result && result.image_base64) {
             resultsOutput.innerHTML = ''; // Clear text
             const img = document.createElement('img');
             img.src = `data:image/png;base64,${result.image_base64}`;
             img.alt = payload.prompt;
             resultsOutput.appendChild(img);
             // Display metadata separately
             const metadata = {...result};
             delete metadata.image_base64; // Don't show base64 string here
             resultsMetadata.textContent = `Metadata:\n${JSON.stringify(metadata, null, 2)}`;

        } else if (stream && result.type === 'final'){
             // For streaming, final chunk might have metadata
             if(result.metadata) {
                resultsMetadata.textContent = `Final Metadata:\n${JSON.stringify(result.metadata, null, 2)}`;
             }
        } else if (!stream) {
             // For non-streaming TTT or other types returning text/json
             if (typeof result === 'string') {
                 resultsOutput.textContent = result;
             } else if (typeof result === 'object') {
                 resultsOutput.textContent = '// JSON Result //'; // Indicate JSON
                 resultsMetadata.textContent = `Result:\n${JSON.stringify(result, null, 2)}`;
             } else {
                 resultsOutput.textContent = String(result); // Fallback
             }
        }
    };

    const errorCb = (error) => {
        setStatus(generateStatus, `Generation failed: ${error.message}`, true);
        setLoadingState(false);
    };

    // --- Make the call ---
    client.generate(payload, streamCb, finalCb, errorCb);

}


// --- Event Listeners ---
connectButton.addEventListener('click', initializeClientAndLoadData);
bindingSelect.addEventListener('change', loadModelsForBinding);
refreshModelsButton.addEventListener('click', loadModelsForBinding);
generateButton.addEventListener('click', handleGeneration);

// Toggle optional inputs
addParamsButton.addEventListener('click', () => {
    paramsContainer.style.display = paramsContainer.style.display === 'none' ? 'block' : 'none';
});
addExtraDataButton.addEventListener('click', () => {
    extraDataContainer.style.display = extraDataContainer.style.display === 'none' ? 'block' : 'none';
});


// --- Initial Load ---
// Optionally load data on page load if URL/Key are preset
// initializeClientAndLoadData();