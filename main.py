import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"

import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max

MAIN_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = MAIN_DIR.parent
DATABASE_PATH = ASSIGNMENT_DIR / "database_recordings"
QUERYSET_PATH = ASSIGNMENT_DIR / "query_recordings"
FINGERPRINT_PATH = MAIN_DIR / "fingerprints"
OUTPUT_PATH = MAIN_DIR / "output.txt"

# Major sources for this file:
# Week 1 / Week 2 / Week 3 lab and tutorial material for coding style and signal-processing structure.
# Week 9 Lab 5 Original notebook for STFT spectrograms and peak picking.
# FMP C7S1 Audio Identification notebook/page for constellation maps
# Wang (2003) An Industrial-Strength Audio Search Algorithm for landmark hashes and offset voting
# FMP C7S3 Evaluation page for Top-k and MAP style evaluation


# Step 1a - Set the main settings for the fingerprinting system
def makeConfig(**overrides):
    """Creates the main configuration dictionary used across the full pipeline."""
    config = {
        "sr": 22050, "n_fft": 2048, "hop_length": 512, "win_length": 2048,
        "window": "hann", "fmin": 80.0, "fmax": 5000.0,
        "peak_method": "maximum_filter", "threshold_db": 30.0,
        "threshold_rel": 0.05, "min_distance": 10, "dist_freq": 8, "dist_time": 8,
        "peaks_per_second": 40, "fanout": 15, "min_time_gap": 1,
        "target_time_dist": 80, "target_freq_dist": 120,
        "freq_bin_size": 4, "time_bin_size": 2, "database_file": "fingerprints.pkl",
    }
    config.update(overrides)
    return config

# Step 1b - Put the database parts into one dictionary
def makeDatabase(config, index, tracks):
    """Packs the saved database objects into one dictionary."""
    return {"config": config, "index": index, "tracks": tracks}

# Default parameters used by the fingerprinting system
DEFAULT_CONFIG = makeConfig()


# Ranking evaluation helper for the correct database answer.
def queryGroundTruth(queryName):
    """Get the correct database filename from one query filename."""
    return f"{queryName.split('-snippet-')[0]}.wav"


# Ranking evaluation using Top-1, Top-3, and MAP@3.
def evaluateRankings(queryNames, rankings):
    """Compute Top-1, Top-3, and MAP@3 from the ranked retrieval results."""
    total = 0
    top1 = 0
    top3 = 0
    averagePrecision = 0.0
    rankCounts = {"rank1": 0, "rank2": 0, "rank3": 0, "miss": 0}

    for queryName, rankedTracks in zip(queryNames, rankings):
        total += 1
        truth = queryGroundTruth(queryName)
        try:
            rank = rankedTracks.index(truth) + 1
        except ValueError:
            rank = None

        if rank == 1:
            top1 += 1
            rankCounts["rank1"] += 1
        elif rank == 2:
            rankCounts["rank2"] += 1
        elif rank == 3:
            rankCounts["rank3"] += 1
        else:
            rankCounts["miss"] += 1

        if rank is not None and rank <= 3:
            top3 += 1
            averagePrecision += 1.0 / rank

    if total == 0:
        return {"top1": 0.0, "top3": 0.0, "map": 0.0, "rank_counts": rankCounts}

    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "map": averagePrecision / total,
        "rank_counts": rankCounts,
    }


# Step 2a - Load one audio file and turn it into mono.
def loadAudio(audioPath, config=DEFAULT_CONFIG):
    """Returns one mono waveform at the working sample rate."""
    x, sr = librosa.load(str(audioPath), sr=config["sr"], mono=True)
    return np.asarray(x, dtype=np.float32)


# Step 2b - Build the spectrogram from one audio file.
def computeSpectrogram(audioPath, config=DEFAULT_CONFIG):
    """ Reference:
LAB: Week 9 Lab 5 Original notebook.
STYLE: Week 2 Tutorial STFT examples.
Builds the spectrogram front end used before peak picking. """
    # Load audio
    y = loadAudio(audioPath, config=config)
    sr = config["sr"]

    # Compute and plot STFT spectrogram
    D = np.abs(librosa.stft(y, n_fft=config["n_fft"], window=config["window"],
    win_length=config["win_length"], hop_length=config["hop_length"],))
    freq = librosa.fft_frequencies(sr=sr, n_fft=config["n_fft"])
    mask = np.logical_and(freq >= config["fmin"], freq <= config["fmax"])
    D = D[mask, :]
    return librosa.amplitude_to_db(D, ref=np.max)


# Step 3a - Find peaks with peak_local_max.
def pickPeaksPeakLocalMax(D, min_distance=10, threshold_rel=0.05):
    """ Reference:
LAB: Week 9 Lab 5 Original notebook.
Peak picking is done with peak_local_max on the log spectrogram. """
    coordinates = peak_local_max(np.log(D + 1e-10), min_distance=min_distance,
    threshold_rel=threshold_rel,exclude_border=False,)
    return coordinates

# Step 3b - Create the constellation map from the spectrogram.
def computeConstellationMap(Y, dist_freq=7, dist_time=7, thresh=0.01):
    """ Reference:
FMP: C7S1 Audio Identification.
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html
Builds the boolean constellation map using the local-neighborhood maximum rule. """
    result = maximum_filter(Y, size=[2 * dist_freq + 1, 2 * dist_time + 1], mode="nearest",)
    return np.logical_and(Y == result, result >= thresh)


# Step 3c - Keep a reasonable number of strong peaks.
def limitPeakDensity(coordinates, spectrogramDb, config=DEFAULT_CONFIG):
    """ Reffrence:
PAPER: Wang (2003) An Industrial-Strength Audio Search Algorithm.
Keeps a controlled number of strong peaks per second before landmark pairing. """
    if len(coordinates) == 0:
        return np.empty((0, 3), dtype=int)

    k = coordinates[:, 0]
    n = coordinates[:, 1]
    values = spectrogramDb[k, n]
    order = np.argsort(values)[::-1]

    framesPerSecond = config["sr"] / config["hop_length"]
    peaksPerSecond = Counter()
    peaks = []

    for idx in order:
        n0 = int(n[idx])
        k0 = int(k[idx])
        secondBucket = int(n0 / framesPerSecond)
        if peaksPerSecond[secondBucket] >= config["peaks_per_second"]:
            continue

        peaksPerSecond[secondBucket] += 1
        peaks.append((n0, k0, int(round(values[idx]))))

    peaks.sort(key=lambda item: (item[0], item[1]))
    return np.asarray(peaks, dtype=int)

# Step 3d - Choose the peaks we want to use for fingerprinting.
def detectPeaks(spectrogramDb, config=DEFAULT_CONFIG):
    """ Reference:
   LAB: Week 9 Lab 5 Original notebook.
FMP: C7S1 Audio Identification.
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html
Selects peak coordinates using either the lab-style local-max method or the FMP constellation-map method. """
    if config["peak_method"] == "peak_local_max":
        D = spectrogramDb - spectrogramDb.min()
        coordinates = pickPeaksPeakLocalMax(D, min_distance=config["min_distance"],
        threshold_rel=config["threshold_rel"],)
    elif config["peak_method"] == "maximum_filter":
        Cmap = computeConstellationMap(spectrogramDb,dist_freq=config["dist_freq"],
        dist_time=config["dist_time"], thresh=spectrogramDb.max() - config["threshold_db"],)
        coordinates = (np.argwhere(Cmap) if np.any(Cmap) else np.empty((0, 2), dtype=int))
    else:
        raise ValueError(f"Unsupported peak_method: {config['peak_method']}")

    return limitPeakDensity(coordinates, spectrogramDb, config)


# Step 4a - Turn the peaks into landmark hashes.
def hashesFromPeaks(peaks, config=DEFAULT_CONFIG):
    """ Reference:
PAPER: Wang (2003) An Industrial-Strength Audio Search Algorithm.
Converts peaks into landmark hashes using anchor points, target points, and time differences. """
    hashes = defaultdict(list)
    if len(peaks) == 0:
        return dict(hashes)

    n = peaks[:, 0]
    k = peaks[:, 1]
    for anchorIndex, (n0, k0) in enumerate(zip(n, k)):
        linkedPoints = 0
        for targetIndex in range(anchorIndex + 1, len(peaks)):
            n1 = int(n[targetIndex])
            deltaN = n1 - int(n0)
            if deltaN < config["min_time_gap"]:
                continue
            if deltaN > config["target_time_dist"]:
                break

            k1 = int(k[targetIndex])
            if abs(k1 - int(k0)) > config["target_freq_dist"]:
                continue

            hashKey = (
                int(k0) // config["freq_bin_size"],
                int(k1) // config["freq_bin_size"],
                int(deltaN) // config["time_bin_size"],
            )
            hashes[hashKey].append(int(n0))
            linkedPoints += 1
            if linkedPoints >= config["fanout"]:
                break

    return dict(hashes)


# Step 4b - Make the fingerprint for one audio file.
def fingerprintFile(audioPath, config=DEFAULT_CONFIG):
    """Run the full fingerprint extraction path for one recording."""
    spectrogramDb = computeSpectrogram(audioPath, config)
    peaks = detectPeaks(spectrogramDb, config)
    return hashesFromPeaks(peaks, config)

# Step 5a - Build the fingerprint database for all tracks.
def buildDatabase(databasePath, config=DEFAULT_CONFIG):
    """Build the database index of hashes for all recordings."""
    trackPaths = sorted(Path(databasePath).glob("*.wav"))
    index = defaultdict(list)
    trackNames = []

    for trackPath in trackPaths:
        trackNames.append(trackPath.name)
        trackHashes = fingerprintFile(trackPath, config)
        for hashKey, anchorTimes in trackHashes.items():
            index[hashKey].extend((trackPath.name, anchorTime) for anchorTime in anchorTimes)

    return makeDatabase(config=config, index=dict(index), tracks=trackNames)


# Step 5b - Save the fingerprint database to disk.
def saveDatabase(database, fingerprintPath):
    """Save the fingerprint database to disk"""
    outputDir = Path(fingerprintPath)
    outputDir.mkdir(parents=True, exist_ok=True)
    outputFile = outputDir / database["config"]["database_file"]

    payload = {
        "config": dict(database["config"]),
        "index": database["index"],
        "tracks": database["tracks"],
    }
    with outputFile.open("wb") as handle:
        pickle.dump(payload, handle)
    return outputFile

# Step 5c - Load the fingerprint database from disk.
def loadDatabase(fingerprintPath):
    """Load the saved fingerprint database from disk."""
    outputFile = Path(fingerprintPath) / DEFAULT_CONFIG["database_file"]
    if not outputFile.exists():
        raise FileNotFoundError(f"No Wang fingerprint database found at {outputFile}")

    with outputFile.open("rb") as handle:
        payload = pickle.load(handle)

    config = makeConfig(**payload["config"])
    return makeDatabase(config=config, index=payload["index"], tracks=payload["tracks"])


# Step 5d - Match a query to the database and rank the tracks.
def searchDatabase(database, queryHashes, top_k=3):
    """ Reference:
PAPER: Wang (2003) An Industrial-Strength Audio Search Algorithm.
Matches query hashes to database hashes and ranks tracks using repeated offset voting. """
    votesByTrack = defaultdict(Counter)

    for hashKey, queryTimes in queryHashes.items():
        matches = database["index"].get(hashKey)
        if not matches:
            continue
        for queryTime in queryTimes:
            for trackName, databaseTime in matches:
                votesByTrack[trackName][databaseTime - queryTime] += 1

    if not votesByTrack:
        return database["tracks"][:top_k]

    rankedTracks = []
    for trackName, offsetCounter in votesByTrack.items():
        bestOffset, bestVotes = offsetCounter.most_common(1)[0]
        totalVotes = sum(offsetCounter.values())
        rankedTracks.append((trackName, bestVotes, totalVotes, -abs(bestOffset)))

    rankedTracks.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
    winners = [trackName for trackName, *_ in rankedTracks]
    if len(winners) < top_k:
        winnerSet = set(winners)
        winners.extend(track for track in database["tracks"] if track not in winnerSet)
    return winners[:top_k]


# Step 6a - Build and save the fingerprint database.
def fingerprintBuilder(database_path, fingerprint_path):
    """Build and save the fingerprint database."""
    database = buildDatabase(database_path, DEFAULT_CONFIG)
    savedPath = saveDatabase(database, fingerprint_path)
    print(f"Saved fingerprint database to {savedPath}")
    print(f"Indexed {len(database['tracks'])} tracks")


# Step 6b - Run the query files and write the results.
def audioIdentification(queryset_path, fingerprint_path, output_path):
    """Write each query filename together with its ranked matches."""
    database = loadDatabase(fingerprint_path)
    queryFiles = sorted(Path(queryset_path).glob("*.wav"))
    outputFile = Path(output_path)
    outputFile.parent.mkdir(parents=True, exist_ok=True)

    queryNames = []
    rankings = []
    with outputFile.open("w", encoding="utf-8") as handle:
        for queryFile in queryFiles:
            queryHashes = fingerprintFile(queryFile, database["config"])
            rankedTracks = searchDatabase(database, queryHashes, top_k=3)
            queryNames.append(queryFile.name)
            rankings.append(rankedTracks)
            handle.write("\t".join([queryFile.name, *rankedTracks]) + "\n")

    metrics = evaluateRankings(queryNames, rankings)
    metrics["query_names"] = queryNames
    metrics["rankings"] = rankings
    print(f"Wrote results to {outputFile}")
    print(f"Processed {len(queryNames)} queries")
    print(f"Top-1 accuracy: {metrics['top1']:.4f}")
    print(f"Top-3 accuracy: {metrics['top3']:.4f}")
    print(f"MAP@3 (equiv. to MRR@3 here): {metrics['map']:.4f}")
    return metrics


# Step 6c - Run the full system with the default folders.
def runAll(rebuildFingerprints=False):
    """Run the default pipeline."""
    if rebuildFingerprints:
        fingerprintBuilder(DATABASE_PATH, FINGERPRINT_PATH)
    return audioIdentification(QUERYSET_PATH, FINGERPRINT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    runAll(rebuildFingerprints=False)
