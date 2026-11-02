# DiffDock : question scientifique et protocole proposé

Recherche vérifiée le **1 novembre 2026**. Cette note définit un protocole et ses limites ; elle ne rapporte pas de nouveaux résultats GPU. Les résultats ne devront être ajoutés qu'avec les manifests, sorties brutes et journaux correspondants.

## Question retenue

**À budget de génération constant, combien d'échecs de sélection de DiffDock-L peut-on corriger avec un contrôle de validité, un score indépendant et un raffinement local, et à quel coût ?**

Le livrable raisonnable est une étude pilote reproductible : un ensemble de complexes publics, quatre variantes utilisant les mêmes candidats, une évaluation sans alignement artificiel du ligand, et un inspecteur permettant de comprendre les échecs. Il ne s'agit ni d'entraîner un nouveau modèle de fondation, ni de revendiquer un nouvel état de l'art.

## Ce que disent les travaux primaires

| Travail | Résultat ou limite utile au projet | Conséquence pratique |
| --- | --- | --- |
| [DiffDock, ICLR 2023](https://arxiv.org/abs/2210.01776) | Sépare génération par diffusion et classement par confiance. La confiance vise la proximité à une pose expérimentale, pas l'affinité. Le récepteur est fixe ; le générateur original représente la protéine par ses Cα, tandis que le modèle de confiance utilise les atomes. | Distinguer échec de génération et échec de classement. Une bonne confiance ne garantit pas une géométrie physique correcte. |
| [DiffDock-L / DockGen, ICLR 2024](https://arxiv.org/abs/2402.18396) | Étudie une généralisation plus exigeante via les domaines protéiques et améliore le modèle par passage à l'échelle et augmentation des données. | Utiliser explicitement DiffDock-L comme baseline moderne. Un petit sous-ensemble PoseBusters ne démontre pas la généralisation à des familles inédites. |
| [PoseBusters, Chemical Science 2024](https://doi.org/10.1039/D3SC04185A) | Montre que le RMSD seul masque des problèmes chimiques, stéréochimiques et stériques ; étudie aussi la minimisation. | Évaluer conjointement précision et validité. « Physiquement plausible » ne signifie pas « synthétisable » ou « actif biologiquement ». |
| [GNINA 1.0, 2021](https://doi.org/10.1186/s13321-021-00522-2) et [GNINA 1.3, 2025](https://doi.org/10.1186/s13321-025-00973-x) | Fournit score de pose CNN, prédiction d'affinité et énergie empirique ; la version 1.3 modifie notamment modèles et données d'entraînement. | Fixer version, modèle et options. Employer `CNNscore` pour le classement annoncé et conserver les autres sorties séparément. |
| [DiffDock-Glide, prépublication 2025](https://www.biorxiv.org/content/10.1101/2025.06.02.657461v1) | Combine DiffDock avec minimisation et classement par Glide. | Le principe d'un pipeline hybride est déjà connu ; éviter toute revendication de nouveauté sur ce seul principe. |
| [Matcha, prépublication, version du 26 octobre 2026](https://arxiv.org/html/2510.14586v4) | Utilise génération, minimisation, contrôles de validité et classement GNINA ; l'annexe I.3 applique ce post-traitement à des baselines, dont DiffDock-L. | L'apport propre doit être la traçabilité et l'analyse des modes d'échec à petit budget, pas l'invention du filtrage ou du reclassement. |
| [RLDiff, Chemical Science 2026](https://doi.org/10.1039/D6SC02655A) | Entraîne par renforcement un modèle dérivé de DiffDock-Pocket pour incorporer précision et validité physique. | Piste future plus coûteuse. Son contexte avec information de poche diffère du docking aveugle retenu ici. |

Le [dépôt officiel DiffDock](https://github.com/gcorso/DiffDock) utilise DiffDock-L par défaut depuis 2024 ; le modèle original est accessible au tag `v1.0`. Le pipeline existant fixe déjà une révision amont et les poids v1.1 : conserver ces références et les empreintes des fichiers dans chaque run. Ne pas appeler indistinctement cette baseline « DiffDock original ».

## Données et budget

Source proposée : [archive PoseBusters sur Zenodo](https://zenodo.org/records/8278563), fichier `posebusters_paper_data.zip`, 53,7 MB, MD5 publié `f004ac7c4e68317b5348497d2bb6bee6`. L'archive contient l'ensemble initial de **428** cas ; l'article final utilise **308** cas après exclusion de contacts cristallins. Employer la [liste finale officielle](https://github.com/maabuu/posebusters/files/14516485/posebusters_pdb_ccd_ids.txt), également [présente dans PoseBench](https://github.com/BioinfoMachineLearning/PoseBench/blob/main/data/posebusters_pdb_ccd_ids.txt). La liste actuellement fournie dans `DiffDock/data/splits/posebusters_benchmark_set_ids.txt` contient encore 428 entrées : elle ne doit pas être utilisée comme alias du jeu 308.

Le [lecteur officiel DiffDock](https://github.com/gcorso/DiffDock/blob/main/datasets/pdbbind.py) attend des répertoires `<PDB>_<CCD>` avec `<ID>_protein.pdb` et `<ID>_ligands.sdf`. Ce SDF peut contenir plusieurs poses cristallographiques du même ligand. Vérifier les fichiers extraits avant de construire le manifest ; le schéma attendu par le lecteur ne remplace pas une inspection de l'archive téléchargée.

Protocole cible, à figer avant la première inférence :

1. Un cas 1A0Q pour vérifier le pipeline, exclu des conclusions scientifiques.
2. Cinq complexes de développement et vingt complexes d'évaluation disjoints, tirés de la liste 308 avec un générateur et une graine enregistrés. Conserver la liste complète ordonnée et les exclusions dans un manifest de données. Ne pas remplacer après coup un complexe parce que le résultat est mauvais.
3. Vingt candidats par complexe et par graine, avec trois graines d'inférence enregistrées. Une seule graine peut servir au pilote initial, mais son résultat doit être présenté comme préliminaire. Ce sont vingt complexes répétés trois fois, pas soixante complexes indépendants.
4. Même récepteur préparé, même ligand et mêmes candidats pour toutes les variantes. Déclarer à l'avance le traitement des eaux, cofacteurs, métaux, états de protonation et structures covalentes. Pour une première étude limitée, exclure les cas nécessitant un traitement spécialisé selon des critères explicites et compter ces exclusions.
5. Ligand de référence conservé séparément pour l'évaluation. Fournir à l'inférence son identité chimique, par exemple un SMILES isomérique, sans utiliser la position cristallographique pour définir une boîte ou une poche. Retirer le ligand à prédire du récepteur.
6. Si plusieurs références de même identité sont présentes, définir avant l'expérience le minimum de RMSD sur **toutes** ces copies dans le repère du récepteur. Enregistrer l'indice de la référence gagnante ; ne pas sélectionner manuellement la copie après avoir vu le résultat.

Un sous-ensemble de vingt complexes permet un diagnostic de faisabilité, pas une conclusion générale sur tous les récepteurs. Une affirmation de généralisation nécessiterait un audit des recouvrements avec les données d'entraînement de DiffDock-L **et** du modèle GNINA, ou un benchmark approprié tel que DockGen.

## Comparaisons contrôlées

| Variante | Coordonnées utilisées | Filtre et classement |
| --- | --- | --- |
| A — confiance | Candidats bruts | Confiance DiffDock décroissante |
| B — validité puis confiance | Les mêmes candidats bruts | Candidats valides, puis confiance décroissante |
| C — validité puis GNINA | Les mêmes candidats bruts | Candidats valides, puis `CNNscore` décroissant |
| D — raffinement puis GNINA | Chaque candidat brut minimisé séparément | Validité après minimisation, puis `CNNscore` décroissant |

A/B isole l'effet du filtre ; B/C celui du score. C/D modifie les coordonnées et peut aussi modifier le score : ce n'est pas une simple expérience de classement. Raffiner tous les candidats dans D permet d'observer le sauvetage éventuel de poses initialement invalides. Conserver leurs identifiants parents. Définir le départage des scores égaux par rang initial puis identifiant stable.

Aucun score utilisant le RMSD à la référence ne participe au classement. L'« oracle » choisissant le meilleur candidat selon la référence sert uniquement à mesurer la marge de progression disponible. Si le filtre ne laisse aucun candidat, enregistrer une abstention et un échec au taux de succès global ; ne pas substituer silencieusement la pose A.

## Métriques et calcul correct

Critère principal : **top-1 avec RMSD des atomes lourds < 2 Å et validité physique**. Rapporter aussi top-1 RMSD seul, validité seule, succès conjoint top-5, succès conjoint de l'oracle du pool, taux d'abstention, erreurs de calcul et temps. Le seuil strict `< 2 Å` est celui retenu dans cette étude, comme dans le papier DiffDock. Le test RMSD de la configuration PoseBusters `redock` utilise `≤ 2 Å` : il n'est pas employé ici et la différence au seuil exact doit rester explicite.

Calculer le RMSD dans le repère du récepteur, avec équivalences atomiques et sans ajuster la pose prédite sur la référence. [RDKit `CalcRMS`](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html) calcule cette distance en place, contrairement à `GetBestRMS`, qui effectue un alignement. Retirer les hydrogènes et vérifier identité, nombre d'atomes, liaisons et stéréochimie avant le calcul. L'appariement automatique de [`CalcRMS`](https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolAlign/AlignMolecules.cpp) n'impose pas la chiralité (`useChirality=false`) : il ne remplace donc pas le contrôle stéréochimique. Fixer et documenter la symétrisation des groupes terminaux conjugués, ainsi que la limite de recherche des correspondances.

Pour chaque complexe, séparer : absence de candidat correct et valide dans le pool ; présence d'un tel candidat mais erreur du classement ; récupération après raffinement. Présenter les comptes bruts avant les pourcentages. Si des intervalles sont calculés, rééchantillonner les complexes de façon appariée, en gardant leurs graines groupées. Ne pas agréger les poses comme des observations indépendantes.

## Interfaces vérifiées et pièges à éviter

### PoseBusters

L'[API officielle](https://posebusters.readthedocs.io/en/latest/api.html) fournit :

```python
from posebusters import PoseBusters

checks = PoseBusters(config="dock", max_workers=0).bust(
    mol_pred="candidate.sdf", mol_cond="receptor.pdb", full_report=False
)
```

La configuration [`dock`](https://github.com/maabuu/posebusters/blob/main/posebusters/config/dock.yml) ne demande pas la pose expérimentale. Compléter ses contrôles par une vérification d'identité et de stéréochimie contre le ligand **d'entrée**. Selon la configuration figée, les contrôles portent notamment sur chargement, connectivité, géométrie, énergie interne et collisions. L'énergie interne utilisée ici est un contrôle de plausibilité, pas une affinité.

La [table retournée](https://github.com/maabuu/posebusters/blob/main/posebusters/posebusters.py) a pour index `file`, `molecule`, `position`. Les noms publics résultent des renommages de configuration, mis en minuscules, avec espaces remplacés par `_` : par exemple `bond_lengths`, `internal_energy`, `minimum_distance_to_protein`. Avec `full_report=False`, seules les sorties de test choisies sont retournées ; `full_report=True` contient aussi des valeurs numériques. Déduire la liste autorisée de la configuration de la version installée, vérifier présence et type booléen, puis considérer un test manquant comme un échec. Ne pas appliquer aveuglément `all()` à un rapport contenant des nombres ou des valeurs manquantes.

La configuration [`redock`](https://github.com/maabuu/posebusters/blob/main/posebusters/config/redock.yml) ajoute notamment identité et RMSD. Son test renommé `RMSD ≤ 2Å` devient `rmsd_≤_2å` dans la table actuelle. Inclure cette colonne dans le filtre de sélection provoquerait une fuite de la vérité terrain. Si cette configuration est utilisée pour l'évaluation, séparer explicitement le module `rmsd` des contrôles de validité. L'[identité PoseBusters](https://github.com/maabuu/posebusters/blob/main/posebusters/modules/identity.py) et sa [normalisation InChI](https://github.com/maabuu/posebusters/blob/main/posebusters/tools/inchi.py) peuvent déduire la stéréochimie des coordonnées : un simple SMILES sérialisé depuis les annotations du fichier n'est pas un remplacement équivalent.

### GNINA

Commandes de base documentées par le [dépôt officiel](https://github.com/gnina/gnina), à adapter à la version binaire figée :

```bash
gnina -r receptor.pdb -l candidate.sdf --score_only -o scored.sdf
gnina -r receptor.pdb -l candidate.sdf --minimize -o refined.sdf
```

La première évalue la pose ; la seconde modifie ses coordonnées localement. Il n'est pas nécessaire de définir une boîte à partir du ligand de référence pour ces opérations. Enregistrer les options complètes de scoring/CNN, version, empreinte du binaire, code retour, stdout/stderr et durée. Ne pas faire dépendre le succès d'un parseur fragile de texte arrondi si les propriétés SDF sont disponibles.

Le [code d'écriture des résultats](https://github.com/gnina/gnina/blob/master/gninasrc/lib/result_info.cpp) distingue `CNNscore` (plus élevé pour le classement retenu), `CNNaffinity`, `minimizedAffinity` (énergie empirique, plus basse) et `minimizedRMSD`. Ce dernier mesure le déplacement par rapport à la pose d'entrée, **pas** l'erreur à la structure cristallographique. `minimizedAffinity` peut également apparaître en mode `score_only` : son nom ne prouve pas qu'une minimisation a eu lieu. Conserver ces propriétés sous leurs noms exacts.

## Corrections nécessaires à la présentation historique

La [fiche RCSB 6AGT](https://www.rcsb.org/structure/6AGT) décrit une **lysyl-ARNt synthétase de Plasmodium falciparum**, pas une sirtuine. Son inhibiteur `9X0` est le *N-(cyclohexylmethyl)-4-oxo-4H-1-benzopyran-2-carboxamide* (`C17H19NO3`). Le SMILES `COc(cc1)ccc1C#N` utilisé dans les essais correspond au 4-méthoxybenzonitrile, qui est une autre molécule. Une pose de cette molécule n'a donc pas de RMSD de redocking valide à `9X0` ; elle constitue seulement une exploration sans pose de référence correspondante.

La confiance DiffDock n'est ni une énergie en kcal/mol, ni une mesure d'affinité. Une corrélation entre cette confiance et un score GNINA compare deux sorties de modèles ; elle ne valide pas une activité biologique. Une valeur de corrélation historique ne doit être reprise comme résultat vérifié qu'après récupération des vecteurs bruts. De même, des confiances identiques une fois arrondies ne prouvent pas que les coordonnées ou les exécutions sont identiques.

## Preuves à conserver pour conclure le projet

- Manifest des données : identifiants, provenance, références, exclusions, préparations et empreintes.
- Manifest de chaque run : commit, poids, environnement, configuration effective, graine, scores en pleine précision, poses et trajectoires réellement sauvegardées.
- Rapport par pose : identité, RMSD, référence retenue, contrôles de validité, score indépendant, identifiant parent après raffinement et erreurs éventuelles.
- Tableau A/B/C/D avec dénominateurs, coûts et échecs ; figures de cas choisis par des critères déclarés, y compris des échecs.
- Inspecteur distinguant pose prédite, référence utilisée seulement pour diagnostic, et trajectoire du sampler. Un schéma pédagogique ou une animation synthétique doit rester identifié comme tel.

Les tests logiciels et un cas de fonctionnement valident l'outillage. Seuls les runs accompagnés de ces preuves permettront de conclure sur les hypothèses scientifiques ; un gain nul ou négatif reste un résultat utile si l'expérience est correctement contrôlée.
