import re
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss(log_file, output_file='loss_curve.png'):
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return

    steps = []
    losses = []
    phases = []
    
    # Regex to match log lines:
    # step=   100  phase=   MATH  loss=  2.3578  lr=1.00e-03  speed=    0.1 step/s
    log_pattern = re.compile(r'step=\s*(\d+)\s+phase=\s*(\w+)\s+loss=\s*([0-9.]+)')

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                step = int(match.group(1))
                phase = match.group(2)
                loss = float(match.group(3))
                
                steps.append(step)
                losses.append(loss)
                phases.append(phase)

    if not steps:
        print(f"No valid log lines found in '{log_file}'.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot the full loss curve
    plt.plot(steps, losses, label='Training Loss', color='#2ca02c', linewidth=2)
    
    # Find phase boundaries and draw vertical lines
    current_phase = phases[0]
    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            # Phase changed
            plt.axvline(x=steps[i], color='gray', linestyle='--', alpha=0.7)
            plt.text(steps[i], max(losses)*0.9, f"  Start {phases[i]}", color='gray', rotation=90)
            current_phase = phases[i]

    plt.title('Curriculum Training Loss', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    print(f"Plot saved successfully to {output_file}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss from log file')
    parser.add_argument('--log', type=str, default='out.txt', help='Path to the training log file (default: out.txt)')
    parser.add_argument('--out', type=str, default='loss_curve.png', help='Path to save the plot (default: loss_curve.png)')
    args = parser.parse_args()
    
    plot_loss(args.log, args.out)
