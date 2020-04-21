# The Peterson/Dolev-Klawe-Rodeh Algorithm 

## Setup:

* Install Microsoft MPI. Go to [installation page](https://www.microsoft.com/en-us/download/details.aspx?id=54607) and download MSMpiSetup.exe. Once downloaded, run the executable and follow the instructions.
  Next, add the path C:\Program Files\Microsoft MPI\bin to the PATH environment variable. You can do this by typing the command:
  ```
  PATH=%PATH%;C:\Program Files\Microsoft MPI\bin
  ```

* Install MPI for Python.
  ```
  pip install -U mpi4py
  ```

* Install Matplotlib.
  ```
  pip install -U matplotlib
  ```
## Run:
  ```
  mpiexec -n {num_of_processes} python network.py random
  ```
  or
  ```
  mpiexec -n {num_of_processes} python network.py from_file {/path/to/file}
  ```
  or
  ```
  # main.py:

  import network

  network.simulate(['from_file', 'input.txt'])
  ```
  ```
  mpiexec -n {num_of_processes} python main.py
  ```

  The file must contain num_of_processes - 1 lines, each of which contains an integer with an identity of a process. The number of processes involved in simulation == num_of_processes - 1, because one process draws all others.
  Despite the fact, that algorithm uses network with a ring topology, for drawing each process uses point-to-point communication with the drawer.