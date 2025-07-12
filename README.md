# 🧬 AI Evolution Simulation

A visual predator–prey simulation built in Python using neural networks and genetic algorithms. Inspired by *Spore*, this project explores emergent behavior in artificial lifeforms through evolution, survival pressure, and energy-based interaction.

![Screenshot Placeholder](#) <!-- Replace with actual image link later -->

---

## 🧠 Features

* Creatures evolve using simple neural networks
* Predators also evolve and must feed to survive
* Realistic energy mechanics and feeding cooldowns
* Mutation and selection drive generational improvement
* Visual simulation using Pygame
* Time control: speed up, slow down, or skip generations
* Stat overlay showing generation, energy, kills, and survivors

---

## 🚀 Getting Started

### 📦 Install dependencies

```bash
pip install pygame numpy
```

### ▶️ Run the simulation

```bash
python evo_sim_feeding.py
```

Use keyboard controls:

* `+` / `-` to increase or decrease simulation speed
* `S` to skip to the next generation

---

## 🧬 How It Works

Each creature and predator has a simple neural network with randomized weights. Over time:

* The best-performing individuals (most food eaten / kills made) are cloned
* Small mutations are introduced to create variation
* Survival pressure leads to emergent behaviors over generations

Predators must *hunt*, *kill*, and *feed* — but must stay still while feeding or risk losing their energy bonus.

---

## 🔮 Planned Features

* Visual display of brain inputs/outputs per agent
* Graphs showing fitness trends over time
* Saving/loading top-performing agents
* Port to Unreal Engine for 3D simulation
* Interactive debugging tools for brains

---

## 📁 Project Structure

```
evo_sim_feeding.py   # Main simulation script
README.md            # Project overview (this file)
assets/              # Placeholder for future images or exports
```

---

## 🤝 Contributing

Pull requests, issues, and feature ideas are welcome!

---

## 📜 License

MIT License — do what you want, credit appreciated.

---

## 🧠 Author

[stitched-dev](https://github.com/stitched-dev)
