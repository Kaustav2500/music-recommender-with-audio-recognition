const API_URL = '';

function showLandingPage() {
    document.getElementById('landingPage').style.display = 'block';
    document.getElementById('inputPage').style.display = 'none';
    document.getElementById('resultsPage').style.display = 'none';
}

function showInputPage() {
    document.getElementById('landingPage').style.display = 'none';
    document.getElementById('inputPage').style.display = 'block';
    document.getElementById('resultsPage').style.display = 'none';
    document.getElementById('songInput').value = '';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('loadingMessage').style.display = 'none';
}

function showResultsPage() {
    document.getElementById('landingPage').style.display = 'none';
    document.getElementById('inputPage').style.display = 'none';
    document.getElementById('resultsPage').style.display = 'block';
}

async function searchSong() {
    const songName = document.getElementById('songInput').value.trim();
    const errorMsg = document.getElementById('errorMessage');
    const loadingMsg = document.getElementById('loadingMessage');
    const searchBtn = document.querySelector('.search-btn');

    if (!songName) {
        errorMsg.textContent = 'Please enter a song name';
        errorMsg.style.display = 'block';
        loadingMsg.style.display = 'none';
        return;
    }

    // show loading state
    errorMsg.style.display = 'none';
    loadingMsg.style.display = 'block';
    searchBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ song_name: songName })
        });

        const data = await response.json();

        if (response.ok && data.recommendations && data.recommendations.length > 0) {
            displayResults(data);
            showResultsPage();
        } else {
            errorMsg.textContent = data.error || 'Song not found. Please try another song.';
            errorMsg.style.display = 'block';
            loadingMsg.style.display = 'none';
        }
    } catch (error) {
        console.error('Error:', error);
        errorMsg.textContent = 'Failed to connect to server. Please make sure the backend is running.';
        errorMsg.style.display = 'block';
        loadingMsg.style.display = 'none';
    } finally {
        searchBtn.disabled = false;
    }
}

function displayResults(data) {
    const queryInfo = document.getElementById('queryInfo');
    queryInfo.textContent = `Showing recommendations for: ${data.query_song}`;

    const container = document.getElementById('recommendationsList');
    container.innerHTML = data.recommendations.map((song, index) => `
        <div class="song-item">
            <div class="song-name">SONG ${index + 1}</div>
            <div class="song-details">
                Song Name - ${song.name}<br>
                Similarity Score - ${song.similarity}
            </div>
            <div class="song-year">Rank<br>#${index + 1}</div>
        </div>
    `).join('');
}

// allow enter key to search
document.addEventListener('DOMContentLoaded', function() {
    const inputField = document.getElementById('songInput');
    if (inputField) {
        inputField.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchSong();
            }
        });
    }
});