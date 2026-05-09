from pathlib import Path
from collections import Counter

MAIN_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = MAIN_DIR.parent
DATABASE_PATH = ASSIGNMENT_DIR / "database_recordings"
QUERYSET_PATH = ASSIGNMENT_DIR / "query_recordings"
FINGERPRINT_PATH = MAIN_DIR / "fingerprints"
OUTPUT_PATH = MAIN_DIR / "output.txt"

# Major sources for this file:
# FMP C7S3 Evaluation page for Top-k and MAP style evaluation.
# FMP C7S1 Audio Identification page for constellation-map and matching ideas.
# Wang (2003) An Industrial-Strength Audio Search Algorithm for offset voting.
# Week 9 Lab 5 for spectrogram and peak-plot style.


# Type 1a - Ranking evaluation helper for the correct answer.
def queryGroundTruth(queryName):
   """Get the correct database filename from one query filename."""
   return f"{queryName.split('-snippet-')[0]}.wav"

# Type 1b - Ranking evaluation using Top-1, Top-3, and MAP@3.
def evaluateRankings(queryNames, rankings):
   """ Reference:
FMP: C7S3 Evaluation.
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S3_Evaluation.html

Compute Top-1, Top-3, and MAP@3 from the ranked retrieval results. """
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


# Type 2 - Runtime and database evaluation.
def benchmarkSystemStats():
   """Measure build time, query time and index size."""
   import time
   import main

   buildStart = time.perf_counter()
   database = main.buildDatabase(DATABASE_PATH, main.DEFAULT_CONFIG)
   buildTime = time.perf_counter() - buildStart
   uniqueHashKeys = len(database["index"])
   totalPostings = sum(len(postings) for postings in database["index"].values())

   trackPaths = sorted(Path(DATABASE_PATH).glob("*.wav"))
   peakCounts = []
   peaksPerSecond = []
   for trackPath in trackPaths:
      signal = main.loadAudio(trackPath, database["config"])
      duration = len(signal) / database["config"]["sr"] if len(signal) else 0.0
      spectrogramDb = main.computeSpectrogram(trackPath, database["config"])
      peaks = main.detectPeaks(spectrogramDb, database["config"])
      peakCounts.append(len(peaks))
      peaksPerSecond.append((len(peaks) / duration) if duration > 0 else 0.0)

   queryPaths = sorted(Path(QUERYSET_PATH).glob("*.wav"))
   queryTimes = []
   for queryPath in queryPaths:
      queryStart = time.perf_counter()
      queryHashes = main.fingerprintFile(queryPath, database["config"])
      main.searchDatabase(database, queryHashes, top_k=3)
      queryTimes.append(time.perf_counter() - queryStart)

   stats = {
      "tracks_indexed": len(database["tracks"]),
      "queries_processed": len(queryTimes),
      "build_time_seconds": buildTime,
      "unique_hash_keys": uniqueHashKeys,
      "total_postings": totalPostings,
      "avg_peaks_per_track": sum(peakCounts) / len(peakCounts) if peakCounts else 0.0,
      "avg_peaks_per_second": sum(peaksPerSecond) / len(peaksPerSecond) if peaksPerSecond else 0.0,
      "total_query_time_seconds": sum(queryTimes),
      "avg_query_time_seconds": sum(queryTimes) / len(queryTimes) if queryTimes else 0.0,
      "min_query_time_seconds": min(queryTimes) if queryTimes else 0.0,
      "max_query_time_seconds": max(queryTimes) if queryTimes else 0.0,
   }

   print("\nRuntime and index statistics")
   for key, value in stats.items():
      if isinstance(value, float):
         print(f"{key}: {value:.4f}")
      else:
         print(f"{key}: {value}")
   return stats


# Type 3a - Visual evaluation helper for spectrogram and peaks.
def plotSpectrogramWithPeaks(audioPath, ax, title, config=None):
   """ Reference:
LAB: Week 9 Lab 5.
FMP: C7S1 Audio Identification.
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html

Plot one spectrogram together with its detected peaks. """
   import main

   if config is None:
      config = main.DEFAULT_CONFIG
   spectrogramDb = main.computeSpectrogram(audioPath, config)
   peaks = main.detectPeaks(spectrogramDb, config)

   ax.imshow(spectrogramDb, origin="lower", aspect="auto", cmap="gray_r")
   if len(peaks): ax.scatter(peaks[:, 0], peaks[:, 1], s=8, color="crimson", alpha=0.75)
   ax.set_title(title)
   ax.set_xlabel("Time frame")
   ax.set_ylabel("Frequency bin")


# Type 3b - Visual evaluation helper for offset votes.
def offsetVotesForTrack(database, queryHashes, trackName):
   """ Reference:
PAPER: Wang (2003) An Industrial-Strength Audio Search Algorithm.
Count the time-offset votes collected for one candidate track. """
   votes = Counter()
   for hashKey, queryTimes in queryHashes.items():
      for matchedTrack, databaseTime in database["index"].get(hashKey, []):
         if matchedTrack != trackName:
            continue
         for queryTime in queryTimes:
            votes[int(databaseTime) - int(queryTime)] += 1
   return votes


# Type 3c - Visual evaluation with one offset-voting example.
def plotOffsetVotingExample(metrics, showPlot=False):
   """ Reference:
   PAPER: Wang (2003) An Industrial-Strength Audio Search Algorithm.
   FMP: C7S1 Audio Identification.
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html

Plot one histogram showing how repeated time offsets vote for the top match. """
   import matplotlib.pyplot as plt
   import main

   database = main.loadDatabase(FINGERPRINT_PATH)
   queryName = None
   topMatch = None

   for name, ranking in zip(metrics["query_names"], metrics["rankings"]):
      if ranking and ranking[0] == queryGroundTruth(name):
         queryName = name
         topMatch = ranking[0]
         break

   if queryName is None:
      return

   queryHashes = main.fingerprintFile(QUERYSET_PATH / queryName, database["config"])
   votes = offsetVotesForTrack(database, queryHashes, topMatch).most_common(25)
   if not votes:
      return

   fig, ax = plt.subplots(figsize=(10, 4))
   ax.bar(range(len(votes)), [count for _, count in votes])
   ax.set_xticks(range(len(votes)))
   ax.set_xticklabels([str(offset) for offset, _ in votes], rotation=45, ha="right")
   ax.set_xlabel("Database time frame - query time frame")
   ax.set_ylabel("Hash votes")
   ax.set_title(f"Offset Voting Example\n{queryName} -> {topMatch}")
   plt.tight_layout()
   if showPlot:
      plt.show()


# Type 3d - Visual evaluation with one good and one bad case.
def plotGoodBadCases(metrics, showPlot=False):
   """ Reffrence:
LAB: Week 9 Lab 5.
FMP: C7S1 Audio Identification.
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html

Plot one successful and one failed retrieval case. """
   import matplotlib.pyplot as plt
   import main

   goodCase = None
   badCase = None

   for name, ranking in zip(metrics["query_names"], metrics["rankings"]):
      truth = queryGroundTruth(name)
      if goodCase is None and ranking and ranking[0] == truth:
         goodCase = (name, truth, ranking[0])
      if badCase is None and (truth not in ranking or ranking[0] != truth):
         badCase = (name, truth, ranking[0] if ranking else None)
      if goodCase and badCase:
         break

   if not goodCase or not badCase or badCase[2] is None:
      return

   fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
   plotSpectrogramWithPeaks(QUERYSET_PATH / goodCase[0], axes[0, 0], f"Successful query\n{goodCase[0]}")
   plotSpectrogramWithPeaks(DATABASE_PATH / goodCase[1], axes[0, 1], f"Correct database track\n{goodCase[1]}")
   plotSpectrogramWithPeaks(QUERYSET_PATH / badCase[0], axes[1, 0], f"Misranked query\n{badCase[0]}")
   plotSpectrogramWithPeaks(DATABASE_PATH / badCase[2], axes[1, 1], f"Wrong top-1 prediction\n{badCase[2]}")
   fig.suptitle("Good and Bad Retrieval Case Studies")
   if showPlot:
      plt.show()

# Type 4 - Run all evaluation code and optional plots.
def runEvaluation(rebuildFingerprints=False, plot=True, benchmark=False):
   """Run the main identification system and optional analysis helpers."""
   import main

   if rebuildFingerprints:
      main.fingerprintBuilder(DATABASE_PATH, FINGERPRINT_PATH)
   metrics = main.audioIdentification(QUERYSET_PATH, FINGERPRINT_PATH, OUTPUT_PATH)
   if benchmark:
      benchmarkSystemStats()
   if plot:
      plotOffsetVotingExample(metrics, showPlot=True)
      plotGoodBadCases(metrics, showPlot=True)
   return metrics


if __name__ == "__main__":
   runEvaluation(rebuildFingerprints=False, plot=True, benchmark=False)
