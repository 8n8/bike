This is a project I did to teach me more about neural networks in Python, as preparation for a robot navigation project.

The main idea behind the project is for a neural network to learn from the choices a human driver makes when navigating a world full of unpredictable moving obstacles.  The main part of this program is a little game where the player controls a robot moving through a world with these properties:

1. It is perfectly flat and smooth.
2. The only objects in it (except for the robot controlled by the player) are fixed-width vertical cylinders.
3. These cylinders move at a constant velocity.

The aim of the game is to keep as close to the given velocity as possible, without crashing into any of the obstacles.  Two views are shown to the player, an arial view of the obstacles around the robot, and four images, showing the view from each of four cameras mounted on the robot.  The cameras are arranged at 90 degrees to each other: forward, back, left and right.

The camera images and player navigation choices are stacked up in memory as the game progresses, and saved to disk when the player chooses.

The next part of the program is a trainer, which uses the data saved from the game to train a convolutional neural network.

The final part of the program is an automatic player of the game, using the trained neural network.

# Installation

This installation should work in Ubuntu Linux 18.04.1 LTS.  I didn't actually start with a clean Ubuntu install, so you might find that there is something you need that is not mentioned here.  It shouldn't be too hard to work out though.

1. You will need Python3.6.  It is likely already installed - check with ```python3.6 --version```.
2. Install tk for the GUI - the game uses Tkinter.  This is available in Ubuntu with ```sudo apt install tk```.
3. Check that pip (the Python package manager) is installed with ```pip3 --version```.  (Don't forget the '3' after pip, because you need the version for Python 3, not Python 2.)  If not, install it with ```sudo apt install python3-pip```.
4. Install virtualenv with ```pip3 install virtualenv```.  (Used for isolating project dependencies, so you don't have to mess with the global Python packages to get something to run.)
1. Clone this repository with ```git clone git@bitbucket.org:5-o/bike.git``` and ```cd``` into it.
5. Set up a new virtual environment for the project with ```python3.6 -m virtualenv venv```.
6. Enter the virtual environment with ```source venv/bin/activate```.
7. Install the dependencies with ```pip install -r requirements.txt```.
8. If you wish, run the tests with ```pytest```.

# Playing the game

Run the command ```python run_game_manual.py```.

The top part of the window shows the images from the four cameras mounted on the robot: left, top, right and back.  The lower part is an arial view of the simulated world, showing the robot as a large red dot and the obstacles as small black dots.

There are two arrows drawn on the robot.  The red one is the actual velocity of the robot and the black is the desired velocity.  The aim of the game is to keep the two arrows the same length and pointing in the same direction.  You will notice that the world is centered on the robot, so for example when you turn right, it looks like the whole world rotates to the left instead.

Keyboard controls:

+ Up arrow - go faster
+ Down arrow - go slower
+ Left arrow - turn left
+ Right arrow - turn right
+ letter 's' - Save data to disk for training.  Don't leave it too long before you do this, because if you crash into an obstacle, all the data since the last save is lost.

The more data the better.

# Training the network

The data from playing the game is saved into the directory 'game_data'.  Run the trainer with the command ```python run_trainer.py```.  The trainer does 10 passes over the data.  The trained model is saved in the file 'nav_net.h5'.

# Let the neural net play the game

Run the command ```python run_game_auto.py```.  The game GUI window will open and the robot will drive automatically.
