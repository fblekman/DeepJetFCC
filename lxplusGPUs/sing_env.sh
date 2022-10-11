#!/bin/bash

singularity run --bind /eos --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
#singularity exec --bind /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest some_command.sh
#singularity exec --bind /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest pwd; ls;
#singularity shell --bind /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest "source some_command.sh && ls && /bin/bash -norc"
