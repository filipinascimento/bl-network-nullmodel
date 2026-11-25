[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.1-blue.svg)](https://doi.org/10.25663/brainlife.app.277)

# Network Null-Models

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Generates an ensemble of null-model networks (Erdos-Renyi, Barabasi-Albert or configuration model) matching an input network, using igraph.**

Reads a network (JGF) and, for each network in it, generates count random realisations with the chosen null model; weights can be ignored, sampled or averaged from the original. Use it to build null distributions for network measures computed on connectomes.

| | |
|---|---|
| 📥 **Inputs** | `generic/network` |
| 📤 **Outputs** | `generic/network` |
| 🐳 **Container** | `filsilva/cxnetwork:0.2.0` |
| ⚙️ **Resources** | 1 CPUs |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.277](https://doi.org/10.25663/brainlife.app.277) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `network` | `generic/network` | yes | Input network |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `output` | `generic/network` | Network file containing the realizations of the null-model. |

<details>
<summary><b>⚙️ Parameters</b> (4)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `count` | number | `100` | Number of null model samples to be generated. |
| `method` | enum | `random` | Model to generate the null samples. - random: ER network with same number of nodes and edges as the original network. - barabasi: Scale-free model with similar number of nodes and edges as the original network. - configuration: Generates a network with same (or similar) degree sequence as the original. Options: `random`, `barabasi`, `configuration`. |
| `weights` | enum | `ignore` | How weights should be handled? Options: `ignore`, `sample`, `average`. |
| `configuration-method` | enum | `simple` | If configuration model is selected, this selects the method to be used. - Simple: The generated null model may have self loops and multiple edges. - No self-loops and multiple edges: Avoids creating self-loops and multiple edges. (Much slower than "simple") - Monte-Carlo: Uses Monte-Carlo method to randomize the networks. Based on the Fabien Viger's method. Options: `simple`, `no_multiple`, `vl`. |

</details>

📚 **References used by this App:**
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)
- Barabasi A-L. Network science. Cambridge University Press (2016).

<!-- brainlife-app:end -->

Generates a esemble of networks according to null models that try to reproduce the data. Erdos reyni (random), Barabási-Albert and Configuration model are implemented.

### Authors
- [Filipi N. Silva](https://filipinascimento.github.io)

### Contributors
- [Franco Pestilli](https://liberalarts.utexas.edu/psychology/faculty/fp4834)


### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your publications and code reusing this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)


### Citations
1. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

2. Barabási, Albert-László. Network science. Cambridge university press, 2016.
Harvard [http://barabasi.com/networksciencebook/](http://barabasi.com/networksciencebook/)


## Running the App 

### On Brainlife.io

You can submit this App online at [https://doi.org/10.25663/brainlife.app.277](https://doi.org/10.25663/brainlife.app.277) via the "Execute" tab.

### Running Locally (on your machine)
Singularity is required to run the package locally.

1. git clone this repo.

```bash
git clone <repository URL>
cd <repository PATH>
```

2. Inside the cloned directory, edit `config-sample.json` with your data or use the provided data.

3. Rename `config-sample.json` to `config.json` .

```bash
mv config-sample.json config.json
```

4. Launch the App by executing `main`

```bash
./main
```

### Sample Datasets

A sample dataset is provided in folder `data` and `config-sample.json`

## Output

The output is a `network` datatype containing just the generated realizations of the null model.


### Dependencies

This App only requires [singularity](https://www.sylabs.io/singularity/) to run. If you don't have singularity, you will need to install the python packages defined in `environment.yml`, then you can run the code directly from python using:  

```bash
./main.py config.json
```

