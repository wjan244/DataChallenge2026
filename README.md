# Datachallenge WEC


## Datasets

### Celeba
```
!wget https://stratus.binets.fr/s/jFHKwZbmmnKcBF4/download -O celeba.zip
unzip celeba.zip -d data/
```


## Speed test
Webp vs jpeg

with batch 128: jpeg: 04:53 / webp: 4:50
removed the isntallati

## Meeting notes

### 26 MAI


#### William
- Contraintes: ne fonctionne que sur un transformer
- Problème: ne fonctionne pas sur le cluster de télécom: conflit d'environement: 
  - @Corentin: y regarder
  - @Corentin critique et relecture
  - @Corentin MLFow/dagshub/weighand biais
- ML flow avec URL sur dagshub

#### Andrew
- Dino et finetuner LP:
  - sur mac: 3hr / 5 epochs et loss diminuait encore
- Tout est trop lourd pour ce qu'on veut faire
- Passe 3D et reprojette en 2D: focus (algo EM) and gan qui essaie de générer la partie manquante et adversariale 
- Hydre à deux tête d'attention: jusqu'à quel point et mettre de l'attention sur un pixel ou regarder par patch
  - Classification en 3D: décors, visage, -> beaucoup moins de paramètres et trois classes: donc si deux forcément le troisième

#### idées
- ellipse de visage et d'occlusion
- Petite app de viz
- Lancer Claude en reflexion max