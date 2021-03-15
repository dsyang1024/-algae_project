import algae_predct as ap
filename = "data"
ap.data_receive(filename)
ap.normalize()
ap.train()