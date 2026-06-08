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


## Optuna results

LP on cls and mean with validation with the wrongly labelled images
```bash
Best score : 0.0053
Best params: {'lp_lr': 0.001522914201547978, 'lp_hidden': 512, 'lp_dropout': 0.21954354652814423, 'lp_weight_decay': 0.033680618365000574, 'lp_loss_alpha': 3.07236135637927, 'lp_loss_beta': 0.2442955094961001, 'smooth_alpha': 5, 'lp_loss': 'PWGLoss'}
````

LP on cls and mean wihout the validation on the worst samples
```bash
Best score : 0.0027
Best params: {'lp_lr': 0.0008699032142533894, 'lp_hidden': 128, 'lp_dropout': 0.04584798620435065, 'lp_weight_decay': 0.05663556977730435, 'lp_loss_alpha': 2.50544967169355, 'lp_loss_beta': 0.28548037714158087, 'smooth_alpha': 85, 'lp_loss': 'PWGLoss'}
```

## CNN on Pathc

embedding job id: 14280870

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