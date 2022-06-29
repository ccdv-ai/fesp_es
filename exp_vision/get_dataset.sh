mkdir -p datasets/oxford-iiit-pet 

wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -P datasets/oxford-iiit-pet
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz -P datasets/oxford-iiit-pet

tar -xvzf datasets/oxford-iiit-pet/images.tar.gz -C datasets/oxford-iiit-pet
tar -xvzf datasets/oxford-iiit-pet/annotations.tar.gz -C datasets/oxford-iiit-pet