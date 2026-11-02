A detailed dataset description, as presented in `dataset.md`.

### Table 1. Description of the Private Power Data
```bash
Table 1 provides information on the data collection scenarios for UAV transmission line inspection, image dimensions, and annotation specifications.  
```

| Item                           | Description                        |
| ------------------------------ | ---------------------------------- |
| Dataset                        | Private Power Data                 |
| Acquisition platform           | UAV                                |
| Acquisition region             | A northern region of China         |
| Acquisition scenario           | Power transmission line inspection |
| Image size used for training   | 640 × 640 pixels                   |
| Annotation type                | Bounding-box annotations           |
| Dataset split (Train:Val:Test) | 7:1:2                              |
| Object scales                  | Small, Medium, Large               |

### Table 2. Number of samples in Private Power Data
```bash
Table 2 presents the numbers of images and instances for the six object categories, together with the training, validation, and test set splits.
```

| Dataset   | Category ID | Category Name | Image Number | Object Number | Train | Test | Val |
| --------- | ----------- | ------------- | ------------ | ------------- | ----- | ---- | --- |
| Pin       | 0           | uninstall     | 1832         | 2262          | 6579  | 1880 | 940 |
| Pin       | 1           | rust          | 1950         | 3981          | —     | —    | —   |
| Pin       | 2           | defect        | 6011         | 7768          | —     | —    | —   |
| Insulator | 3           | burn          | 475          | 670           | 951   | 272  | 137 |
| Insulator | 4           | defect        | 508          | 530           | —     | —    | —   |
| Insulator | 5           | dirty         | 440          | 591           | —     | —    | —   |

*Note: Train, Test, and Val represent the numbers of images in the corresponding subsets for each dataset group.*

### Table 3. Number of samples at different object resolutions in Private Power Data
```bash
Table 3 reports the distribution of object instances across different scales.
```
| Sample Resolution | Small | Medium | Large |
| ----------------- | ----- | ------ | ----- |
| pin-uninstall     | 58    | 1420   | 774   |
| pin-rust          | 546   | 2145   | 741   |
| pin-defect     | 680   | 5618   | 1470  |
| insulator-burn    | —     | —      | 670   |
| insulator-defect  | 5     | 21     | 504   |
| insulator-dirty   | 2     | 3      | 586   |