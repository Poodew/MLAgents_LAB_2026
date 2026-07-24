using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

public class CartpoleAgent : Agent
{
    [SerializeField] CartController controller;

    public override void Initialize()
    {

    }

    public void FixedUpdate()
    {
        RequestDecision();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(controller.cartX);
        sensor.AddObservation(controller.cartVel);
        sensor.AddObservation(controller.poleAngle);
        sensor.AddObservation(controller.poleAngVel);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        int action = actionBuffers.DiscreteActions[0];
        Debug.Log($"Action: {action}");

        float input = action == 0 ? -1f : 1f;
        controller.Move(input);

        AddReward(0.01f);

        if (controller.IsFailed())
        {
            AddReward(-1f);
            EndEpisode();
        }
    }

    public override void OnEpisodeBegin()
    {
        controller.ResetCart();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActions = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.LeftArrow))
            discreteActions[0] = 0;
        
        else if (Input.GetKey(KeyCode.RightArrow))
            discreteActions[0] = 1;
        
    }
}
