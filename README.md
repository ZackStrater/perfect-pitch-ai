## Perfect Pitch AI
# Using Deep Learning to Transcribe Music From Audio

If you have ever tried learning to learn to play an instrument or even if you've played an instrument your whole life, you will know that one of the most difficult problems for casual musicians have is learning new songs.  Players either must try to learn a song by ear, which is an incredibly difficult task that even for professional musicians, or they must procure a written music, which can often be limited in selection or unreliable.  This project is aimed at helping musicians learn the songs they want to learn using deep learning.  The ultimate goal is an AI product that can "listen" to an audio file and instantly transcribe it into an accurate and accesible form of written music.  

# Machine Learning Approach
The major difficulty of this project how do you get a machine learning model to "listen" to music?  The answer is you teach it to "see" music instead.  The advantage of reframing this problem as a visual one is we get to make use of one of the most powerful tools in the modeling arsenel: convolutional neural networks (CNNs).  This type of deep learning model is very powerful at parsing 2D and 3D data and has been used to great effect in the field of computer vision.  To get audio data into a visual format, we can use a fourier transform on the raw audio waveform (via the Librosa library).  This results in a 2D frequency spectrogram that shows what frequencies occur at every point in time during the song.  Here is an example frequency spectrogram from piano audio.  You can see that the resulting image is complex, which is due to the fact that each note contains not only its base frequency, but several other harmonics that occur at higher frequency registers.  This visual format of the music will now be the input data that our CNN will train on to learn what frequencies relate to what notes.

The target data will be a digital written form of music, MIDI, which contains all the information about when each note is pressed, released, and how hard each note is pressed.  In this approach, the model will guess which notes are being played at every moment during the song, which makes this a multilabel classification problem.  At any given time, the model must make a binary prediction for each of the 88 notes on a piano whether or not that note is currently being played or not. 

# Data and Data Preparation 
The data for this project comes from the MAESTRO database, a compilation of 200 hours of classical piano performances from the International Piano-e-Competition.  The database contains synced audio and MIDI files for each performance.  First, I set up a pipeline to convert both the audio and MIDI files into NumPy arrays so they could be fed into our model.  As mentioned above, the audio files were converted to arrays using a short-time Fourier Transform (STFT), which allows us to see which frequencies are present at each point in time.  The result is an audio spectrogram with frequencies on the y axis and time on the x axis.  In this case, I used a mel-spectrogram, which buckets the frequencies into equally sized frequency bins.  This transformation solves two problems: it reduces the dimensionality of the input images (necessary for memory constraints), and it "de-logs" the audio. 


# Results

# Conclusion
