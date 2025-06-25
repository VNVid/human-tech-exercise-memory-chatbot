from user_study_config import STREAM_THOUGHTS, USE_IMAGES

if not USE_IMAGES:
    title = "LegBot"
    description = "Hi there! I’m your leg-focused exercise coach. I can explain how different leg exercises work and when to use them."
    example1 = "What are the best exercises for strong legs?"
    example2 = "Recommend an exercise that strengthens the hamstrings without equipment."

elif not STREAM_THOUGHTS:
    title = "ArmBot"
    description = "Hi there! I’m your arm and shoulder expert. I recommend exercises and explain how they help your upper body."
    example1 = "What are the best biceps exercises using dumbbells?"
    example2 = "Suggest some exercises for shoulder strength."

else:
    title = "TorsoBot"
    description = "Hi there! I specialize in torso workouts. I recommend exercises for your core, abs, back and chest and explain how they work."
    example1 = "Suggest exercises that activate abs."
    example2 = "Find an exercise that strengthens the spine for posture."