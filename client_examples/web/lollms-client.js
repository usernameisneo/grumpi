// client_examples/web/lollms-client.js

class LollmsClient {
    constructor(baseUrl, apiKey) {
        if (!baseUrl) {
            throw new Error("Base URL is required for LollmsClient.");
        }
        // Ensure base URL doesn't end with a slash for easier joining
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
        this.apiKey = apiKey;
        this.headers = {
            "Content-Type": "application/json",
        };
        if (this.apiKey) {
            this.headers["X-API-Key"] = this.apiKey;
        } else {
            console.warn("LollmsClient initialized without API Key. Requests might fail if server requires authentication.");
        }
    }

    /**
     * Generic request helper
     * @param {string} method - HTTP method (GET, POST, etc.)
     * @param {string} endpoint - API endpoint path (e.g., '/list_bindings')
     * @param {object|null} [body=null] - Request body for POST/PUT requests
     * @param {boolean} [stream=false] - If true, returns the raw response for streaming
     * @returns {Promise<object|string|Response>} - Parsed JSON, text, or raw Response object
     */
    async _request(method, endpoint, body = null, stream = false) {
        const url = `${this.baseUrl}${endpoint}`;
        const options = {
            method: method,
            headers: { ...this.headers },
        };
        if (body) {
            options.body = JSON.stringify(body);
        }

        try {
            console.debug(`Making ${method} request to ${url}`, body ? `with body:` : '', body ?? '');
            const response = await fetch(url, options);

            if (!response.ok) {
                let errorBody;
                try {
                    errorBody = await response.json(); // Try parsing error detail as JSON
                } catch (e) {
                    errorBody = await response.text(); // Fallback to text
                }
                console.error(`API request failed: ${response.status} ${response.statusText}`, errorBody);
                throw new Error(`HTTP error ${response.status}: ${JSON.stringify(error_detail || errorBody)}`);
            }

            if (stream) {
                console.debug("Returning raw response for streaming.");
                return response; // Return the raw response for stream handling
            }

            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                console.debug("Received JSON response.");
                return await response.json();
            } else if (contentType && contentType.includes("text/plain")) {
                console.debug("Received text response.");
                return await response.text();
            } else {
                console.debug("Received response with unknown content type, returning text.");
                return await response.text(); // Default to text for unknown types
            }
        } catch (error) {
            console.error(`Network or other error during request to ${url}:`, error);
            throw error; // Re-throw the error for the caller to handle
        }
    }

    // --- Public API Methods ---

    async listBindings() {
        return await this._request("GET", "/list_bindings");
    }

    async listPersonalities() {
        return await this._request("GET", "/list_personalities");
    }

    async listAvailableModels(bindingName) {
        if (!bindingName) throw new Error("Binding name is required to list available models.");
        return await this._request("GET", `/list_available_models/${bindingName}`);
    }

    /**
     * Performs generation (text, image, etc.)
     * @param {object} payload - The generation request payload (personality, prompt, model_name, etc.)
     * @param {function} [streamCallback=null] - (chunkData) => void - Called for each streaming chunk ('chunk', 'error', 'info' types)
     * @param {function} [finalCallback=null] - (finalData) => void - Called with the final result (non-streaming) or final stream chunk ('final' type)
     * @param {function} [errorCallback=null] - (error) => void - Called on any error during request or streaming
     */
    async generate(payload, streamCallback = null, finalCallback = null, errorCallback = null) {
        const isStreaming = payload.stream === true;

        if (isStreaming && (!streamCallback || !finalCallback)) {
            console.warn("Streaming requested, but streamCallback or finalCallback not provided. Use non-streaming or provide callbacks.");
            // Optionally default to non-streaming? Or just proceed and potentially lose data?
            // For now, proceed, but callbacks won't be hit.
        }

        try {
            const response = await this._request("POST", "/generate", payload, isStreaming);

            if (!isStreaming) {
                // Handle non-streaming response
                if (finalCallback) {
                    finalCallback(response);
                } else {
                    console.log("Non-streaming generation completed:", response);
                }
                return response; // Return the final result for non-streaming
            } else {
                // Handle streaming response (Server-Sent Events)
                if (!response.body) {
                    throw new Error("Response body is null, cannot process stream.");
                }
                const reader = response.body
                    .pipeThrough(new TextDecoderStream())
                    .getReader();

                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        console.debug("Stream finished.");
                        // Check if there was any leftover data in buffer (shouldn't happen with correct SSE)
                        if (buffer.trim()) {
                             console.warn("Stream ended with unprocessed data in buffer:", buffer);
                        }
                        break;
                    }

                    buffer += value;
                    // Process buffer line by line for SSE messages (data: JSON\n\n)
                    let boundary = buffer.indexOf('\n\n');
                    while (boundary !== -1) {
                        const message = buffer.substring(0, boundary);
                        buffer = buffer.substring(boundary + 2); // Move buffer past the message + \n\n
                        boundary = buffer.indexOf('\n\n'); // Check for next message

                        if (message.startsWith('data:')) {
                            const jsonData = message.substring(5).trim(); // Remove 'data:' prefix
                            if (jsonData) {
                                try {
                                    const chunkData = JSON.parse(jsonData);
                                    if (chunkData.type === 'chunk' || chunkData.type === 'info' || chunkData.type === 'error') {
                                        if (streamCallback) streamCallback(chunkData);
                                    } else if (chunkData.type === 'final') {
                                        if (finalCallback) finalCallback(chunkData);
                                    } else {
                                         console.warn("Received unknown chunk type:", chunkData);
                                         if (streamCallback) streamCallback(chunkData); // Still pass it?
                                    }
                                } catch (e) {
                                    console.error("Failed to parse JSON chunk:", jsonData, e);
                                    if (errorCallback) errorCallback(new Error("Failed to parse stream data"));
                                }
                            }
                        } else if (message.trim()){
                             console.warn("Received non-data SSE line:", message);
                        }
                    } // end while boundary
                } // end while reader
            } // end else isStreaming
        } catch (error) {
            console.error("Error during generate call:", error);
            if (errorCallback) {
                errorCallback(error);
            } else {
                // Re-throw if no specific handler
                // throw error;
            }
        }
    }
}