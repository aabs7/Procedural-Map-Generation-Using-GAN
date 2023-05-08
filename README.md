# Procedural-Map-Generation-Using-GAN
The project aims to generate large buildings procedurally using GAN. The number of maps generated is arbitrary.

To see all the targets, run: 
```make help```

To build the program, run:
```make build```

To run all the test inside the test dir, run:
```make test```

#### Generating floor plans
To generate floor plans using GAN, run: 
```make save-map-plots```

After running `make save-map-plots`, the generated floor plans are saved in `resources/maps-image`

#### DCGAN Training
To train on mnist using DCGAN, run:
```make train-mnist```

To train on celeb dataset using DCGAN, run:
```make train-celebrity```

To train on floor-dataset using DCGAN, run:
```make train-floor```

To train all at once, run:
```make train-all```

To visualize how DCGAN is performing, run:
```make visualize-data```
[Note: You have to have tensorboard installed for this]