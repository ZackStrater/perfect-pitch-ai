## Perfect Pitch AI
# Using Deep Learning to Transcribe Music From Audio

If you have ever tried learning to learn to play an instrument or even if you've played an instrument your whole life, you will know that one of the most difficult problems for casual musicians have is learning new songs.  Players either must try to learn a song by ear, which is an incredibly difficult task that even for professional musicians, or they must procure a written music, which can often be limited in selection or unreliable.  This project is aimed at helping musicians learn the songs they want to learn using deep learning.  The ultimate goal is an AI product that can "listen" to an audio file and instantly transcribe it into an accurate and accesible form of written music.  

# Machine Learning Approach
The major difficulty of this project how do you get a machine learning model to "listen" to music?  The answer is you teach it to "see" music instead.  The advantage of reframing this problem as a visual one is we get to make use of one of the most powerful tools in the modeling arsenel: convolutional neural networks (CNNs).  This type of deep learning model is very powerful at parsing 2D and 3D data and has been used to great effect in the field of computer vision.  To get audio data into a visual format, we can use a fourier transform on the raw audio waveform (via the Librosa library).  This results in a 2D frequency spectrogram that shows what frequencies occur at every point in time during the song.  Here is an example frequency spectrogram from piano audio.  You can see that the resulting image is complex, which is due to the fact that each note contains not only its base frequency, but several other harmonics that occur at higher frequency registers.  This visual format of the music will now be the input data that our CNN will train on to learn what frequencies relate to what notes.

The target data will be a digital written form of music, MIDI, which contains all the information about when each note is pressed, released, and how hard each note is pressed.  In this approach, the model will guess which notes are being played at every moment during the song, which makes this a multilabel classification problem.  At any given time, the model must make a binary prediction for each of the 88 notes on a piano whether or not that note is currently being played or not. 

# Data and Data Preparation 
The data for this project comes from the MAESTRO database, a compilation of 200 hours of classical piano performances from the International Piano-e-Competition.  The database contains synced audio and MIDI files for each performance.  First, I set up a pipeline to convert both the audio and MIDI files into NumPy arrays so they could be fed into our model.  As mentioned above, the audio files were converted to arrays using a short-time Fourier Transform (STFT), which allows us to see which frequencies are present at each point in time.  The result is an audio spectrogram with frequencies on the Y axis and time on the X axis.  In this case, a Mel-frequency spectrogram was used, which buckets the frequencies into equally sized frequency bins.  This transformation solves two problems: it reduces the dimensionality of the input images (necessary for memory constraints), and it "de-logs" the natural audio scale.  Notes are naturally more bunched up in the lower frequencies and are more spread out in the higher registers (for example, a half step is ~2 hz at the low end of the piano and ~200 hz at the high end.   The Mel-frequency scale allows us to produce a spectrogram that is spaced evenly across the frequency spectrum, which should in turn help our model recognize frequency patterns regardless of whether they are high or low  notes.

The target data was prepared from the MIDI information (what notes are being played at what times).  The result was a 2D array with 88 rows (corresponding to the 88 keys on the piano) where 1 represents a key being pressed at that moment in time and 0 means the key is not being pressed.  Finally the array had to be resized in the X direction in order to make the time ticks equal to the time ticks of the audio spectrogram.

The initial strategy in this project was to take a window of the audio spectrogram (as a grayscale 2D image) and have the model guess just one vertical slice of the MIDI output (i.e. predict what notes are being played during that slice of time).  Each paired set of audio and MIDI files were then cut up into equal sized windows and a slice of the MIDI window was saved as the target data for that audio window.  The length of the audio window could then be adjusted to find the optimal size for the network.  With all the data prepared, we could then proceed to testing the convolutional neural network.

# Results

| Winow Size  | Frames Left | Frames Right  | Modification 1 | Modification 2         | Eval Metrics: | Accuracy | Precision | Recall | F1 Score |
| ----------- | ----------- | ------------- | -------------- | ---------------        | ---           | -------- | --------- | ------ | -------- |
| 60          |         50  |            9  |  128 mels      |              -         |               |    0.33  |     0.64  |   0.44 |     0.52 |
| 60          |         50  |            9  |   **200 mels** |              -         |               |    0.33  |     0.63  |   0.46 |     0.53 |
| 60          |         50  |            9  |   200 mels     |**Filters: 96, 96, 48, 48**|            |    0.32  |     0.63  |   0.39 |     0.48 |
| 60          |         50  |            9  |   200 mels     |**Filters:64, 64, 128, 128**|           |    0.30  |     0.63  |   0.38 |     0.47 |
| 60          |     **9**   |        **50** |   200 mels     |              -         |               |    0.33  |     0.65  |   0.42 |     0.51 |

| 60          |     **10**  |        **9**  |   200 mels     |              -         |               |    0.36  |     0.71  |   0.47 |     0.57 |
| 60          |         50  |            9  |   200 mels     |  **Batch Size = 64**   |               |    0.38  |     0.71  |   0.52 |     0.60 |
| 60          |         50  |            9  |   200 mels     |  **Batch Size = 96**   |               |    0.36  |     0.68  |   0.58 |     0.63 |
| 10          |     **5**   |       **4**   |   200 mels     |      Batch Size = 96   |               |    0.37  |     0.71  |   0.56 |     0.63 |





# Conclusion
